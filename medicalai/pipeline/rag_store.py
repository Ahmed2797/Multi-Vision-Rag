import uuid
import os
from base64 import b64decode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_classic.storage.in_memory import InMemoryStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from IPython.display import display, Image as IPImage
from medicalai.chatmodel import openai_embedding
from langchain_core.messages import HumanMessage




class RAGStore:
    def __init__(
        self,
        index_name="vision-rag-v1",
        cloud="aws",
        region="us-east-1",
        dimension=1536,
    ):
        self.store = InMemoryStore()
        self.embeddings = openai_embedding()
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not set in environment variables.")

        pc = Pinecone(api_key=pinecone_api_key)
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        self.vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embeddings,
        )
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
        # 1. This line checks Pinecone AND loads from Docstore
        docs = self.retriever.invoke(question)
        
        # 2. Separate text from base64 images
        parsed = self.parse_docs(docs, self.store)
        
        # 3. Build a multimodal message for a Vision-capable LLM
        content = []
        
        # Add text context
        context_text = "\n".join([doc.page_content for doc in parsed["texts"]])
        content.append({
            "type": "text", 
            "text": f"Use the following text and images to answer.\nContext: {context_text}\nQuestion: {question}"
        })
        
        # Add the actual image data so the LLM can "see" it
        for b64_img in parsed["images"]:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })

        # 4. Send the full multimodal payload to the LLM
        response = llm.invoke([HumanMessage(content=content)])
        
        return response.content, parsed
        

    @staticmethod
    def parse_docs(docs,store):
        images, texts = [], []
        for doc in docs:
            doc_id = doc.metadata.get("doc_id")
            if not doc_id:
                # Fallback for docs ingested directly into Pinecone without docstore linkage.
                texts.append(doc)
                continue
            original = store.mget([doc_id])[0]
            if not original:
                texts.append(doc)
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
