import os
from typing import TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI

# ===============================
# LLM Setup
# ===============================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# ===============================
# TypedDict Schema
# ===============================
class VerifierResult(TypedDict):
    verdict: Literal["ok", "hallucination"]
    confidence: float

# LLM augmented with structured output parsing (TypedDict ile)
llm_verifier = llm.with_structured_output(VerifierResult)

# ===============================
# Verifier Agent
# ===============================
def run_verifier(query: str, answer: str, context: str) -> VerifierResult:
    """
    Verifier Agent: checks if answer is grounded in context.
    Returns VerifierResult TypedDict.
    """
    resp: VerifierResult = llm_verifier.invoke(f"""
    You are a strict verifier.
    Task: Decide if the answer is grounded in the provided documentation context.

    Query: {query}

    Answer:
    {answer}

    Context:
    {context[:2000]}

    Rules:
    - verdict = "ok" if the answer is fully supported by the context.
    - verdict = "hallucination" if the answer contains unsupported or invented info.
    - confidence must be between 0.0 and 1.0.
    """)
    return resp