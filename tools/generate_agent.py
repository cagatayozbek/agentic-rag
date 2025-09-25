# tools/generate_agent.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langsmith import traceable

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# ===================================
# Low-level function
# ===================================
def run_generate(query: str, context: str, citations: list) -> dict:
    """
    Generate code based on user request and retrieved context.
    """
    if not context.strip():
        return {
            "code": "# Bilgi bulunamadı: İlgili context boş.",
            "citations": [],
        }

    # --- Code generation prompt ---
    prompt = f"""
    You are a coding assistant. 
    Generate Python code that solves the user's request using only the provided context if possible. 
    If the answer cannot be found in the context, still try to generate reasonable code but clearly mark it as "⚠️ speculative".
    
    Context:
    {context}

    User request: {query}

    Return only the code, no explanations.
    """

    resp = llm.invoke(prompt)
    code = resp.content.strip()

   

    return {"code": code, "citations": citations}


# ===================================
# Tool Wrapper
# ===================================
@traceable
@tool
def generate_tool(query: str, context: str, citations: list) -> dict:
    """
    Tool: Generate Python code based on context retrieved from the RAG system.
    """
    return run_generate(query, context, citations)