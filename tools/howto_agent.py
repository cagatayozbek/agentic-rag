import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langsmith import traceable

# ===============================
# LLM Setup
# ===============================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# ===============================
# Low-level function
# ===============================
def run_howto(query: str, context: str, citations: list) -> dict:
    """
    Generate a step-by-step or checklist style answer using retrieved context.
    """
    if not context.strip():
        return {
            "answer": "Bu soruya adım adım yanıt veremem çünkü ilgili bilgi tabanında bulunmuyor.",
            "citations": [],
        }

    # --- Prompt ---
    prompt = f"""
    You are an assistant that writes step-by-step instructions or checklists 
    using only the provided context. 
    If the answer cannot be found in the context, say "I don't know".

    Format the output as:
    1. Step one
    2. Step two
    3. Step three

    Context:
    {context}

    Question: {query}
    Step-by-step Answer:
    """
    resp = llm.invoke(prompt)
    answer = resp.content.strip()




    return {"answer": answer, "citations": citations}

# ===============================
# Tool Wrapper
# ===============================
@traceable
@tool
def howto_tool(query: str, context: str, citations: list) -> dict:
    """
    Tool: Generate step-by-step instructions (How-To) based on retrieved context.
    """
    return run_howto(query, context, citations)