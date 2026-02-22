from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from medicalai.chatmodel import openai_chat_model


# -------------------------------
# LLM Summarizer
# -------------------------------
class LLMSummarizer:
    def __init__(self):
        self.llm = openai_chat_model()

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
