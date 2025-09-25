import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
import cohere
import dotenv
from langchain_core.tools import tool
from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI

dotenv.load_dotenv()

# ===============================
# LLM for Query Optimization
# ===============================
llm_opt = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

def optimize_query(query: str) -> str:
    """
    Use LLM to rewrite/optimize the query for better retrieval.
    """
    prompt = f"""
    You are a query optimization assistant.

    We have a local vector database built from documentation of:
    - LangChain
    - LangGraph
    - LangSmith

    Your task: Rewrite the following user query into a concise, technical **search query**
    that will retrieve the most relevant chunks from this documentation.

    Guidelines:
    - Focus only on concepts, functions, classes, and usage details from these docs.
    - Remove irrelevant words, personal pronouns, or conversational style.
    - Always output in **English**.
    - Do NOT invent new terms. Stay consistent with the documentation vocabulary.
    - Keep it short (5–15 words).

    User query: {query}

    Optimized query:
    """
    resp = llm_opt.invoke(prompt)
    print("🔍 Optimized Query:", resp.content.strip())
    return resp.content.strip()


# ===============================
# Semantic Search (FAISS only)
# ===============================
def semantic_search(query, top_k=20):
    """
    Semantic retrieval (FAISS only).
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")

    FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
    DOCSTORE_PATH = os.path.join(DATA_DIR, "docstore.json")

    model = SentenceTransformer("BAAI/bge-m3")

    with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
        docstore = json.load(f)

    global_ids = [int(k) for k in docstore.keys()]

    index = faiss.read_index(FAISS_INDEX_PATH)
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb.astype("float32"), k=top_k * 2)

    faiss_scores = {global_ids[i]: float(D[0][rank]) for rank, i in enumerate(I[0]) if i != -1}
    if faiss_scores:
        max_faiss = max(faiss_scores.values())
        if max_faiss > 0:
            faiss_scores = {k: v / max_faiss for k, v in faiss_scores.items()}

    ranked = sorted(faiss_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for gid, score in ranked:
        doc = docstore[str(gid)]
        results.append({
            "global_chunk_id": gid,
            "score": round(score, 4),
            "title": doc.get("title", ""),
            "source": doc.get("source", ""),
            "content": doc.get("content", ""),
        })
    return results


# ===============================
# Semantic Search + Cohere Rerank (with Query Optimization)
# ===============================
@tool
@traceable(run_type="tool", name="Retriever Tool")
def hybrid_search_with_rerank(query: str, top_k: int = 10, rerank: bool = True):
    """
    Semantic retrieval (FAISS only) + Cohere Rerank with LLM query optimization.
    """
    # 1. Optimize query first
    optimized = optimize_query(query)

    # 2. Run semantic search
    candidates = semantic_search(optimized, top_k=top_k * 2)

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    if rerank:
        documents = [c["content"] for c in candidates]

        response = co.rerank(
            model="rerank-english-v3.0",
            query=optimized,
            documents=documents,
            top_n=top_k,
        )

        final = []
        for r in response.results:
            doc = candidates[r.index]
            final.append({
                "global_chunk_id": doc["global_chunk_id"],
                "score": round(r.relevance_score, 4),
                "title": doc["title"],
                "source": doc["source"],
                "content": doc["content"],
            })
        return final
    else:
        return candidates[:top_k]