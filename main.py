from medicalai.pipeline.llm_summary import LLMSummarizer
from medicalai.pipeline.pdf_process import PDFProcessor
from medicalai.pipeline.rag_store import RAGStore
from medicalai.chatmodel import openai_chat_model
from medicalai.prompt import summary_prompt_text,vision_prompt_text
# -------------------------------
# Usage Example
# -------------------------------
PDF_PATH = "medicalai/data/attention.pdf"

# Load & extract
processor = PDFProcessor(PDF_PATH)
processor.partition_pdf()
texts, tables = processor.extract_texts_tables()
images = processor.extract_images_base64()

summarizer = LLMSummarizer()
text_summaries = summarizer.summarize_texts([t.text for t in texts], summary_prompt_text)
table_summaries = summarizer.summarize_texts([t.metadata.text_as_html for t in tables], summary_prompt_text)
image_summaries = summarizer.summarize_images(images, vision_prompt_text)

# Store in RAG
rag_store = RAGStore()
for s, t in zip(text_summaries, texts):
    rag_store.add_document(s, t, "text")
for s, t in zip(table_summaries, tables):
    rag_store.add_document(s, t.metadata.text_as_html, "table")
for s, b64 in zip(image_summaries, images):
    rag_store.add_document(s, b64, "image")

# Query

llm = openai_chat_model()
question = "Explain the multi-head attention diagram"
answer, context = rag_store.query(question, llm)

# Display
rag_store.show_images(context["images"])
print("\n=== ANSWER ===\n")
print(answer)

def response(query: str):
    # Initialize LLM
    llm = openai_chat_model()  # your wrapper to create ChatOpenAI

    # Query RAG
    answer, context = rag_store.query(query, llm)

    # Show answer
    print("ANSWER : ",answer)

    # Show images if there are any
    if context.get("images"):
        print(f"Found {len(context['images'])} image(s) in context:")
        rag_store.show_images(context["images"])
    else:
        print("No images found in context.")


