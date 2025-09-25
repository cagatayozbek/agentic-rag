from langchain_google_genai import ChatGoogleGenerativeAI
import os
import dotenv
dotenv.load_dotenv()
# Gemini LLM (LangChain wrapper)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ["GEMINI_API_KEY"]
)

def generate_answer(query: str, context: str):
    prompt = f"""You are a helpful assistant.
User Question: {query}
Retrieved Context: {context}

Answer the question using the context.
"""
    resp = llm.invoke(prompt)
    return resp.content