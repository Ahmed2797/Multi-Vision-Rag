import os
import glob
from typing import List, Tuple
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from medicalai.pipeline.pdf_process import PDFProcessor


def ensure_index(pc: Pinecone, index_name: str, dimension: int, cloud: str, region: str) -> None:
    if index_name in pc.list_indexes().names():
        return
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )


def load_documents_with_pdf_processor(data_path: str) -> Tuple[List[Document], int]:
    pdf_files = sorted(glob.glob(os.path.join(data_path, "**", "*.pdf"), recursive=True))
    documents: List[Document] = []
    total_images = 0

    for pdf_path in pdf_files:
        processor = PDFProcessor(pdf_path=pdf_path)
        processor.partition_pdf()
        texts, tables = processor.extract_texts_tables()
        images = processor.extract_images_base64()
        total_images += len(images)

        for t in texts:
            content = getattr(t, "text", "")
            if content:
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": pdf_path, "type": "text"},
                    )
                )

        for tb in tables:
            content = getattr(tb.metadata, "text_as_html", None) or getattr(tb, "text", "")
            if content:
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": pdf_path, "type": "table"},
                    )
                )

        if images:
            for idx, _ in enumerate(images, start=1):
                documents.append(
                    Document(
                        page_content=f"Image {idx} extracted from {os.path.basename(pdf_path)}",
                        metadata={"source": pdf_path, "type": "image", "image_index": idx},
                    )
                )

    return documents, total_images


def ingest_documents(
    data_path: str = "medicalai/data",
    index_name: str = "vision-rag",
    embedding_model: str = "text-embedding-3-small",
    dimension: int = 1536,
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
    load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not pinecone_api_key:
        raise ValueError("Missing PINECONE_API_KEY in environment variables.")
    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment variables.")
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    pc = Pinecone(api_key=pinecone_api_key)
    ensure_index(pc, index_name=index_name, dimension=dimension, cloud=cloud, region=region)

    documents, image_count = load_documents_with_pdf_processor(data_path=data_path)
    if not documents:
        print(f"No extractable PDF content found under: {data_path}")
        return

    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=openai_api_key)
    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name,
    )
    print(
        f"Ingestion complete: {len(documents)} documents indexed in Pinecone index '{index_name}'. "
        f"Detected images: {image_count}."
    )


if __name__ == "__main__":
    ingest_documents()
