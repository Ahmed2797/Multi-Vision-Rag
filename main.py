from functools import lru_cache

from medicalai.chatmodel import openai_chat_model
from medicalai.pipeline.llm_summary import LLMSummarizer
from medicalai.pipeline.pdf_process import PDFProcessor
from medicalai.pipeline.rag_store import RAGStore
from medicalai.prompt import summary_prompt_text, vision_prompt_text

PDF_PATH = "medicalai/data/attention.pdf"


@lru_cache(maxsize=1)
def get_rag_store() -> RAGStore:
    """Build and cache the local RAG store once per process."""

    # Load & extract
    processor = PDFProcessor(PDF_PATH)
    processor.partition_pdf()
    texts, tables, images = processor.extract_elements()


    # Initialize your refactored summarizer
    # GPT-4.1-nano is excellent for this due to speed and cost
    summarizer = LLMSummarizer(model="gpt-4.1-nano")

    # Define Prompts
    summary_prompt_text = "Produce a concise, factual summary. Preserve key facts and numbers. Output ONLY the summary text."
    vision_prompt_text = "Describe the image precisely. Do not speculate."

    # --- TEXT SUMMARIZATION ---
    # We use a simple list comprehension, but only if 'texts' isn't empty
    text_summaries = []
    if texts:
        # If texts are already strings from processor.extract_elements()
        text_summaries = summarizer.summarize_texts(texts, summary_prompt_text)

    # --- TABLE SUMMARIZATION ---
    table_summaries = []
    if tables:
        # Ensure tables is a list of HTML strings
        table_summaries = summarizer.summarize_texts(tables, summary_prompt_text)

    # --- IMAGE SUMMARIZATION ---
    image_summaries = []
    if images:
        # images should be a list of Base64 strings
        image_summaries = summarizer.summarize_images(images, vision_prompt_text)

    print(f"✅ Summarization Complete: {len(text_summaries)} texts, {len(table_summaries)} tables, {len(image_summaries)} images.")


    # Store in RAG
    rag_store = RAGStore()
    for s, t in zip(text_summaries, texts):
        rag_store.add_document(s, t, "text")
    for s, t in zip(table_summaries, tables):
        rag_store.add_document(s, t.metadata.text_as_html, "table")
    for s, b64 in zip(image_summaries, images):
        rag_store.add_document(s, b64, "image")

    return rag_store


def response(query: str) -> str:
    llm = openai_chat_model()
    rag_store = get_rag_store()
    answer, context = rag_store.query(query, llm)

    print("ANSWER:", answer)
    if context.get("images"):
        print(f"Found {len(context['images'])} image(s) in context:")
        rag_store.show_images(context["images"])
    else:
        print("No images found in context.")

    return answer

if __name__ == "__main__":
    question = "Explain the multi-head attention diagram"
    response(query=question)
