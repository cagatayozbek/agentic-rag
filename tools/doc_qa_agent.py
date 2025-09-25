import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# ===================================
# Low-level function
# ===================================
def run_doc_qa(query: str, context: str, citations: list, with_confidence: bool = True) -> dict:
    """
    Answer a factual question using context retrieved from the RAG system.
    The answer should be short and summarized.
    """
    if not context.strip():
        return {
            "answer": "Bu soruya cevap veremem çünkü ilgili bilgi tabanında bulunmuyor.",
            "citations": [],
            "confidence": 0.0
        }

    # --- Generate summarized answer ---
    prompt = f"""
    You are an assistant that answers questions using ONLY the provided context.
    Summarize the information concisely (2-3 sentences max).
    If the answer cannot be found in the context, reply with "I don't know".

    Context:
    {context}

    Question: {query}
    Short Answer:
    """
    resp = llm.invoke(prompt)
    answer = resp.content.strip()

    confidence = 0.0
    if with_confidence:
        judge_prompt = f"""
        You are a judge. Decide confidence between 0.0 and 1.0.
        Question: {query}
        Answer: {answer}
        Context: {context[:2000]}
        Return only a float number between 0 and 1.
        """
        conf_resp = llm.invoke(judge_prompt)
        try:
            confidence = float(conf_resp.content.strip())
        except:
            confidence = 0.5

    return {"answer": answer, "citations": citations, "confidence": confidence}

# ===================================
# Tool Wrapper
# ===================================
@tool
def doc_qa_tool(query: str, context: str, citations: list) -> dict:
    """
    Tool: Answer factual questions based on context retrieved from the RAG system.
    """
    return run_doc_qa(query, context, citations, with_confidence=True)