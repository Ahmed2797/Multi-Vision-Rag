import uuid
from base64 import b64decode
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_classic.storage.in_memory import InMemoryStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.vectorstores import Chroma
from IPython.display import display, Image as IPImage
from medicalai.chatmodel import openai_chat_model,openai_embedding


#  RAG Storage
# -------------------------------
class RAGStore:
    def __init__(self, persist_dir="./chroma_db"):
        self.store = InMemoryStore()
        self.embeddings = openai_embedding()
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
        parsed = self.parse_docs(docs)
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