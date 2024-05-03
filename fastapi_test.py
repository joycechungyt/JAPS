from fastapi import FastAPI
from langchain_google_vertexai import VertexAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv

_ = load_dotenv()

app = FastAPI()

gemini_model = VertexAI(model_name="gemini-1.0-pro-001")

template = """
You are a language tutor, you're having a conversation with a human.
Answer at the same CEFR level as you determine of the human.

{prev_conv}

Human: {question}
AI:"""
prompt = PromptTemplate.from_template(template)

conversationBuffer = []

chain = prompt | gemini_model

class Query(BaseModel):
    user_id: str
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    global conversationBuffer
    conversationBuffer = conversationBuffer[-10:]
    convo_context = ""
    for idx, conv in enumerate(conversationBuffer):
        if idx % 2 == 0:
            convo_context += f"Human: {conv}\n"
        else:
            convo_context += f"AI: {conv}\n"
    print(convo_context)
    response = chain.invoke({"prev_conv": convo_context, "question": query.question})
    conversationBuffer.append(f"{query.question}")
    conversationBuffer.append(f"{response}")
    return {"message": response}
