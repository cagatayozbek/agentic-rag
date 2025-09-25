from typing_extensions import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import dotenv

dotenv.load_dotenv()

# ===============================
# LLM Setup
# ===============================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# ===============================
# Structured Output Schema
# ===============================
class Plan(BaseModel):
    tool: Literal["answer","generate","explain", "none"] = Field(
        ..., description="Which tool should handle the query. Use 'none' if the query is outside the RAG knowledge base."
    )
    path: Literal["fast", "slow", "none"] = Field(
        ..., description="Execution path. Use 'none' if no tool should be used."
    )

# LLM augmented with structured output
router = llm.with_structured_output(Plan)

# ===============================
# Planner Function
# ===============================
def plan_query(query: str) -> dict:
    """
    LLM-based planner that decides which tool & path to use for a query.
    """
    decision = router.invoke(
    f"""
    You are a routing assistant for the LangChain ecosystem (LangChain, LangGraph, LangSmith). 
    Your ONLY task is to select the correct tool and execution path.

    ❌ Never try to answer the query yourself.  
    ❌ Never guess or add explanations.  
    ✅ Always return exactly one tool and one path.  

    If the question is OUTSIDE this domain (personal info, current weather, user's name, chit-chat, jokes, music, etc.), 
    set tool="none" and path="none". The system will then inform the user it cannot answer.

    ----------------
    Available Tools:
    - answer → factual Q&A, definitions, explanations ("what/why/when") ,s tep-by-step instructions ("how/nasıl") from LangChain docs.
    - generate → code generation requests ("bana şunu yapan kodu ver").
    - explain → explaining/debugging a code snippet ("bu kod ne yapıyor?", "neden böyle?").
    - none → query is outside LangChain ecosystem.
    
    Available Paths:
    - fast → direct factual answer is enough (doc_qa, generate, simple explain).
    - slow → reasoning or multi-step guidance required (howto, detailed explain).
    - none → when tool="none".
    ----------------

    Routing Rules:
    - Factual, LangChain-related → tool="answer", path="fast".
    - Instructional, step-by-step ("how/nasıl") → tool="answer", path="slow".
    - Code generation ("give me code") → tool="generate", path="fast".
    - Code explanation/debugging → tool="explain", path="slow".
    - Anything else (not LangChain ecosystem) → tool="none", path="none".

    User query: {query}
    """
)
    
    return {"tool": decision.tool, "path": decision.path}