import os
import base64
from typing import List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import your custom modules
from medicalai.chatmodel import openai_chat_model
from medicalai.pipeline.rag_store import RAGStore



# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("🚀 System Starting...")
#     try:
#         llm = openai_chat_model()
#         rag_store = RAGStore()
#         print("✅ Models and Store loaded.")
#     except Exception as e:
#         print(f"❌ Startup Error: {e}")


app = FastAPI(title="VisionRag API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    images: Optional[List[str]] = None

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request_data: QueryRequest):
    llm = openai_chat_model()
    rag_store = RAGStore()

    if not llm or not rag_store:
        raise HTTPException(status_code=503, detail="AI Models not ready.")
    
    # 1. Execute Query
    answer, context = rag_store.query(request_data.query, llm)
    
    retrieved_images = context["images"]
    print(context)
    print(retrieved_images)

    return {
        "answer": answer, 
        "images": retrieved_images
    }
# --- Execution ---
# Run via: uvicorn app:app --reload



# raw_images = context
    # print(context)
    # # 2. Convert images to clean Base64 strings
    # clean_images = []
    # for img in raw_images:
    #     try:
    #         if isinstance(img, bytes):
    #             # Convert raw bytes to base64 string
    #             img_str = base64.b64encode(img).decode('utf-8')
    #         else:
    #             # If it's already a string, make sure it's clean
    #             img_str = str(img).strip()
    #             # Remove Python's b'...' prefix if it accidentally got stringified
    #             if img_str.startswith("b'") or img_str.startswith('b"'):
    #                 img_str = img_str[2:-1]
            
    #         clean_images.append(img_str)
    #     except Exception as e:
    #         print(f"Error processing image data: {e}")

    # print(f"--- Query: {request_data.query} ---")
    # print(f"Images found: {len(clean_images)}")