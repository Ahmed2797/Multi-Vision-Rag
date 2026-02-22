from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
from medicalai.chatmodel import openai_chat_model
from medicalai.pipeline.rag_store import RAGStore

app = FastAPI(title="VisionRag API")


app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# -------------------------------
# Request & Response Schemas
# -------------------------------
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    images: Optional[List[str]] = None  # base64 strings


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query", response_model=QueryResponse)
def query_pdf(request_data: QueryRequest):
    query_text = request_data.query

    # Create LLM and RAG store
    llm = openai_chat_model()
    rag_store = RAGStore()

    # Query RAG
    answer, context = rag_store.query(query_text, llm)

    # Extract images
    images = context.get("images", [])

    return QueryResponse(answer=answer, images=images)



## uvicorn app:app --reload
