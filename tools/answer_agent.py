import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

def run_answer(query: str, context: str, citations: list, mode: str = "qa") -> dict:
    """
    Unified Answer Agent.
    mode = "qa"     → factual short answer
    mode = "howto"  → step-by-step instructions
    """
    if not context.strip():
        return {
            "answer": "Bu soruya cevap veremem çünkü ilgili bilgi tabanında bulunmuyor.",
            "citations": [],
        
        }

    if mode == "howto":
        style = "Provide a clear step-by-step guide or checklist."
    else:
        style = "Provide a short factual answer."

    prompt = f"""
    You are an assistant for LangChain ecosystem questions.
    Answer strictly using the provided context.

    Context:
    {context}

    Question: {query}

    Instructions:
    - {style}
    - If the answer is not in the context, reply with: "I don't know."
    """

    resp = llm.invoke(prompt)
    answer = resp.content.strip()

 

    return {"answer": answer, "citations": citations}

@tool
def answer_tool(query: str, context: str, citations: list, mode: str = "qa") -> dict:
    """
    Tool: Unified Answer Agent.
    - mode="qa" → short factual answer
    - mode="howto" → step-by-step instructions
    """
    return run_answer(query, context, citations, mode)