from fastapi import FastAPI
from langchain_google_vertexai import VertexAI
# from langchain.conversation import ConversationChain, ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

_ = load_dotenv()

app = FastAPI()

gemini_model = VertexAI(model_name="gemini-1.0-pro-001")

template = """
Question: {question}

Answer: Let's think step by step.
"""
prompt = PromptTemplate.from_template(template)

chain = prompt | gemini_model

@app.get("/")
async def root(query: str):
    response = gemini_model.invoke(query)
    return {"message": response}