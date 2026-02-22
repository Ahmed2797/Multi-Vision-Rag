from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def openai_chat_model(model_name:str="gpt-4.1-nano"):
    
    chat_model = ChatOpenAI(
    model= model_name, ## "gpt-4.1-nano", ## "gpt-4o-mini",  ## "gpt-5-nano",  # OpenAI chat model
    temperature=0.4, 
    max_tokens=500
    )
    
    return chat_model

def openai_embedding(model_name:str='text-embedding-3-small'):
    embeddings = OpenAIEmbeddings(
        model=model_name, 
        api_key=OPENAI_API_KEY
        )
    return embeddings

