from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Literal
from tools.planner import plan_query
from tools.retriever import hybrid_search_with_rerank
from tools.generate_agent import run_generate  # Generate core function
from tools.explain_agent import run_explain    # Explain core function
from tools.answer_agent import run_answer      # Answer core function
from tools.verifier_agent import run_verifier
from langsmith import Client
import os
import dotenv

dotenv.load_dotenv()
client = Client()

print(f"✅ LangSmith tracing aktif! Proje: {os.getenv('LANGSMITH_PROJECT')}")

# ===============================
# State (SADELEŞTİRİLMİŞ)
# ===============================
class PipelineState(TypedDict, total=False):
    # Zorunlu giriş
    query: str

    # Planner çıktıları
    tool: Literal["answer", "generate", "explain", "none"]
    path: Literal["fast", "slow"]

    # Retrieval çıktıları
    context: str
    citations: List[dict]

    # Agent çıktıları (duruma göre biri veya ikisi)
    answer: str
    code: str
    confidence: float

    # Verifier çıktısı
    verdict: Literal["ok", "hallucination"]

# ===============================
# Planner Node
# ===============================
def planner_node(state: PipelineState):
    decision = plan_query(state["query"])
    return {"tool": decision["tool"], "path": decision["path"]}

# ===============================
# Retrieval Node
# ===============================
def retrieval_node(state: PipelineState):
    results = hybrid_search_with_rerank.invoke({"query": state["query"]})
    context = "\n".join([r["content"] for r in results])
    return {"context": context, "citations": results}

# ===============================
# Doc QA Node
# ===============================
def answer_node(state: PipelineState):
    mode = "qa" if state["path"] == "fast" else "howto"
    result = run_answer(
        state["query"],
        state.get("context", ""),
        state.get("citations", []),
        mode=mode,

    )
    return {
        "answer": result["answer"],
        "citations": result["citations"],
     
    }

# ===============================
# Generate Node
# ===============================
def generate_node(state: PipelineState):
    result = run_generate(
        state["query"],
        state.get("context", ""),
        state.get("citations", []),

    )
    return {
        "code": result["code"],
        "citations": result["citations"],

    }

# ===============================
# Explain Node
# ===============================
def explain_node(state: PipelineState):
    result = run_explain(
        state["query"], state.get("context", ""), True
    )
    return {
        "answer": result["answer"],
        "citations": result["citations"],

    }

# ===============================
# Verifier Node
# ===============================
def verifier_node(state: PipelineState):
    result = run_verifier(
        query=state["query"],
        answer=state.get("answer") or state.get("code", ""),
        context=state.get("context", "")
    )
    return {
        "verdict": result["verdict"],
        "confidence": result["confidence"]
    }

# ===============================
# Fallback Node (domain dışı)
# ===============================
def fallback_node(state: PipelineState):
    return {
        "answer": "Ben sadece LangChain, LangGraph ve LangSmith ekosistemi ile ilgili sorulara yanıt verebilirim.",

    }

# ===============================
# Graph
# ===============================
graph = StateGraph(PipelineState)

graph.add_node("planner", planner_node)
graph.add_node("retrieval", retrieval_node)
graph.add_node("answer", answer_node)
graph.add_node("generate", generate_node)
graph.add_node("explain", explain_node)
graph.add_node("fallback", fallback_node)
graph.add_node("verifier", verifier_node)

graph.set_entry_point("planner")

# Planner sonrası routing
def route_after_planner(state: PipelineState):
    if state["tool"] == "none":
        return "fallback"
    elif state["tool"] in ["answer", "generate", "explain"]:
        return "retrieval"
    return "fallback"

graph.add_conditional_edges(
    "planner", route_after_planner, ["retrieval", "fallback"]
)

# Retrieval sonrası routing
def route_after_retrieval(state: PipelineState):
    if state["tool"] == "answer":
        return "answer"
    elif state["tool"] == "generate":
        return "generate"
    elif state["tool"] == "explain":
        return "explain"

graph.add_conditional_edges(
    "retrieval", route_after_retrieval, ["answer", "generate", "explain"]
)

# Her ana agent node -> verifier
graph.add_edge("answer", "verifier")
graph.add_edge("generate", "verifier")
graph.add_edge("explain", "verifier")

# Finish points (DİKKAT: Son çağrı geçerli kalabilir)
# Finish points
graph.set_finish_point("verifier")
graph.add_edge("fallback", END)

app = graph.compile()

"""
png_bytes = app.get_graph().draw_mermaid_png()
with open("graph_diagram.png", "wb") as f:
    f.write(png_bytes)
"""
mermaid_code = app.get_graph().draw_mermaid()
with open("graph.mmd", "w") as f:
    f.write(mermaid_code)

# ===============================
# Test
# ===============================
if __name__ == "__main__":
    out2 = app.invoke({"query": "LangChain ile nasıl agent oluşturum?"})
    print("\n📌 Query:", out2["query"])
    print("📌 Tool:", out2["tool"], "| Path:", out2["path"])
    print("📌 Code:\n", out2.get("code"))
    print("📌 Citations:", out2.get("citations"))
    print("📌 Verifier Verdict:", out2.get("verdict"))
    print("📌 Confidence:", out2.get("confidence"))