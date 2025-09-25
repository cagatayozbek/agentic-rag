import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from tools.retriever import hybrid_search_with_rerank

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# ===================================
# Low-level function
# ===================================
def run_explain(query: str, code_snippet: str) -> dict:
    """
    Explain or debug a code snippet.
    If it uses LangChain/LangGraph APIs, run retrieval to add context.
    """
    # --- Detect if snippet is LangChain-related ---
    is_langchain = any(
        kw in code_snippet.lower()
        for kw in ["langchain", "langgraph", "langsmith"]
    )

    context = ""
    citations = []
    if is_langchain:
        # retrieve docs for explanation
        results = hybrid_search_with_rerank.invoke({"query": query})
        context = "\n".join([r["content"] for r in results])
        citations = results

    # --- Prompt for explanation ---
    prompt = f"""
    You are an assistant that explains code to a junior developer.

    User Question: {query}

    Code Snippet:
    {code_snippet}

    Context (from docs, may be empty):
    {context}

    Instructions:
    - Explain what the code does in clear, simple terms.
    - If it's LangChain/LangGraph code, use the context for accuracy.
    - If it's unrelated, just explain based on the snippet itself.
    - Keep the explanation short and clear.
    """

    resp = llm.invoke(prompt)
    explanation = resp.content.strip()

    return {
        "answer": explanation,
        "citations": citations,

    }

# ===================================
# Tool Wrapper
# ===================================
@tool
def explain_tool(query: str, code_snippet: str) -> dict:
    """
    Tool: Explain or debug a code snippet. 
    Uses RAG context if it's related to LangChain/LangGraph APIs.
    """
    return run_explain(query, code_snippet)