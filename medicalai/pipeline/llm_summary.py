from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from medicalai.chatmodel import openai_chat_model
from typing import List
import os



class LLMSummarizer:
    def __init__(self, model="gpt-4.1-nano", temperature=0.3):
        # API Key should be handled in the env or passed explicitly
        self.llm = ChatOpenAI(
            model=model, 
            temperature=temperature, 
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=500  # Prevents runaway costs on summaries
        )

    def summarize_texts(self, texts: List[str], prompt_template: str, max_concurrency=5):
        prompt = ChatPromptTemplate.from_template(prompt_template)
        # Standard LCEL chain
        chain = prompt | self.llm | StrOutputParser()
        
        # Batching with the correct config format
        # We pass a list of dicts matching the prompt variable '{element}'
        inputs = [{"element": t} for t in texts]
        return chain.batch(inputs, config={"max_concurrency": max_concurrency})

    def summarize_images(self, images_base64: List[str], prompt_template: str, max_concurrency=2):
        vision_prompt = ChatPromptTemplate.from_messages([
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": "data:image/jpeg;base64,{image}",
                            "detail": "low"  # Industry standard for technical diagrams (saves tokens)
                        }
                    }
                ]
            )
        ])
        
        vision_chain = vision_prompt | self.llm | StrOutputParser()
        
        # We pass a list of dicts matching the prompt variable '{image}'
        inputs = [{"image": img} for img in images_base64]
        return vision_chain.batch(inputs, config={"max_concurrency": max_concurrency})