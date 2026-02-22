# -------------------------------
# 1️⃣ Imports
# -------------------------------
import os
import uuid
from base64 import b64decode
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, CompositeElement, Image
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_classic.storage.in_memory import InMemoryStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.vectorstores import Chroma
from IPython.display import display, Image as IPImage

# -------------------------------
# 2️⃣ PDF Loader & Chunker
# -------------------------------
class PDFProcessor:
    def __init__(self, pdf_path, output_path="vision-rag-pdf"):
        self.pdf_path = pdf_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.chunks = []

    def partition_pdf(self):
        self.chunks = partition_pdf(
            filename=self.pdf_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000
        )
        return self.chunks

    def extract_texts_tables(self):
        texts, tables = [], []
        for el in self.chunks:
            if isinstance(el, CompositeElement):
                texts.append(el)
            elif isinstance(el, Table):
                tables.append(el)
        return texts, tables

    def extract_images_base64(self):
        images = []
        for chunk in self.chunks:
            if isinstance(chunk, CompositeElement):
                for el in chunk.metadata.orig_elements:
                    if isinstance(el, Image) and el.metadata.image_base64:
                        images.append(el.metadata.image_base64)
        return images

# -------------------------------
# 3️⃣ LLM Summarizer
# -------------------------------
class LLMSummarizer:
    def __init__(self, api_key, model="gpt-4.1-nano", temperature=0.3):
        self.llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    def summarize_texts(self, texts, prompt_template: str, max_concurrency=3):
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = ({"element": lambda x: x} | prompt | self.llm | StrOutputParser())
        return chain.batch(texts, {"max_concurrency": max_concurrency})

    def summarize_images(self, images, prompt_template: str, max_concurrency=2):
        vision_prompt = ChatPromptTemplate.from_messages([
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}}
                ]
            )
        ])
        vision_chain = ({"image": lambda x: x} | vision_prompt | self.llm | StrOutputParser())
        return vision_chain.batch(images, {"max_concurrency": max_concurrency})

# -------------------------------
# 4️⃣ RAG Storage
# -------------------------------
class RAGStore:
    def __init__(self, api_key, persist_dir="./chroma_db"):
        self.store = InMemoryStore()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        self.vectorstore = Chroma(collection_name="multi_collection",
                                  embedding_function=self.embeddings,
                                  persist_directory=persist_dir)
        self.retriever = MultiVectorRetriever(vectorstore=self.vectorstore,
                                              docstore=self.store,
                                              id_key="doc_id")
        self.retriever.search_kwargs = {"k": 5}

    def add_document(self, summary, original_content, doc_type):
        doc_id = str(uuid.uuid4())
        # 🔹 Vectorstore
        self.vectorstore.add_documents([
            Document(page_content=summary, metadata={"doc_id": doc_id, "type": doc_type})
        ])
        # 🔹 Docstore
        if doc_type == "image":
            content = original_content
        elif hasattr(original_content, "text"):
            content = original_content.text
        else:
            content = str(original_content)
        self.store.mset([(doc_id, Document(page_content=content, metadata={"doc_id": doc_id, "type": doc_type}))])

    def query(self, question, llm):
        docs = self.retriever.invoke(question)
        parsed = self.parse_docs(docs,self.store)
        context_text = "\n".join([doc.page_content for doc in parsed["texts"]])
        prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the context below.
If the answer is not present, say "Not found in the document."

Context:
{context}

Question:
{question}
""")
        messages = prompt.format_prompt(context=context_text, question=question).to_messages()
        response = llm.invoke(messages)
        return response.content, parsed

    @staticmethod
    def parse_docs(docs,store):
        images, texts = [], []
        for doc in docs:
            doc_id = doc.metadata.get("doc_id")
            if not doc_id:
                continue
            original = store.mget([doc_id])[0]
            if not original:
                continue
            if original.metadata["type"] == "image":
                try:
                    b64decode(original.page_content)
                    images.append(original.page_content)
                except Exception:
                    pass
            else:
                texts.append(original)
        return {"images": images, "texts": texts}

    @staticmethod
    def show_images(images):
        for idx, img in enumerate(images, 1):
            print(f"🖼️ Image {idx}")
            display(IPImage(data=b64decode(img)))

# -------------------------------
# 5️⃣ Usage Example
# -------------------------------
PDF_PATH = "medicalai/data/attention.pdf"
OPENAI_API_KEY = os.getenv("PINECONE_API_KEY")

# Load & extract
processor = PDFProcessor(PDF_PATH)
processor.partition_pdf()
texts, tables = processor.extract_texts_tables()
images = processor.extract_images_base64()

# Summarize
summary_prompt_text = "Produce a concise, factual summary. Preserve key facts and numbers. Output ONLY the summary text."
vision_prompt_text = "Describe the image precisely. Do not speculate."

summarizer = LLMSummarizer(api_key=OPENAI_API_KEY)
text_summaries = summarizer.summarize_texts([t.text for t in texts], summary_prompt_text)
table_summaries = summarizer.summarize_texts([t.metadata.text_as_html for t in tables], summary_prompt_text)
image_summaries = summarizer.summarize_images(images, vision_prompt_text)

# Store in RAG
rag_store = RAGStore(api_key=OPENAI_API_KEY)
for s, t in zip(text_summaries, texts):
    rag_store.add_document(s, t, "text")
for s, t in zip(table_summaries, tables):
    rag_store.add_document(s, t.metadata.text_as_html, "table")
for s, b64 in zip(image_summaries, images):
    rag_store.add_document(s, b64, "image")

# Query
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3, api_key=OPENAI_API_KEY)
question = "Explain the multi-head attention diagram"
answer, context = rag_store.query(question, llm)

# Display
rag_store.show_images(context["images"])
print("\n=== ANSWER ===\n")
print(answer)
