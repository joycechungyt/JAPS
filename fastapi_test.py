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

prompt_template = """Your name is Ella and you are a virtual conversational agent. Have a human like, free flowing conversation with a person who is not a native english speaker. Ask back questions to continue the conversation
  Do not respond in more than 50 words. Be polite. Do not bombard with information. Do not address yourself as a virtual agent. Make up information about yourself if personal details are asked.
  Provide brief answers and recommendations based on the question and the given situation.
  SITUATION: {situation}
  Current Conversation:
  {history}
  human: {input}
  AI: """
prompt = PromptTemplate.from_template(prompt_template)

conversationBuffer = []

chain = prompt | gemini_model

class Query(BaseModel):
    user_id: str
    question: str
    situation: str

@app.post("/ask")
async def ask_question(query: Query):
    global conversationBuffer
    if query.situation == "":
        query.situation = "General conversation"
    currentConversationBuffer = conversationBuffer[-20:]
    convo_context = ""
    for idx, conv in enumerate(currentConversationBuffer):
        if idx % 2 == 0:
            convo_context += f"Human: {conv}\n"
        else:
            convo_context += f"AI: {conv}\n"
    print(convo_context)
    response = chain.invoke({"situation": query.situation,
                             "history": convo_context, 
                             "input": query.question})
    conversationBuffer.append(f"{query.question}")
    conversationBuffer.append(f"{response}")
    return {"message": response}
