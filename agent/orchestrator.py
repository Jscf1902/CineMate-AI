import os
import json
import uuid
from datetime import datetime

from ollama import chat

from src.rag.rag_pipeline import rag_retrieve, is_title_query
from src.retrieval.hybrid_search import hybrid_search_faiss


# -------------------------
# PATHS
# -------------------------
BASE_PATH = r"C:\Users\juans\OneDrive\Documentos\Maestria en Ingenieria y Analitica de Datos\Proyecto de Grado\CineMate AI\interactions"
RAG_CONTROL_PATH = os.path.join(BASE_PATH, "rag_control.json")


# -------------------------
# SESSION MANAGEMENT
# -------------------------
def create_session():
    session_id = str(uuid.uuid4())

    data = {
        "session_id": session_id,
        "created_at": str(datetime.now()),
        "messages": [],
        "rag_usage": []
    }

    save_session(session_id, data)

    return session_id


def load_session(session_id):
    path = os.path.join(BASE_PATH, f"session_{session_id}.json")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_session(session_id, data):
    path = os.path.join(BASE_PATH, f"session_{session_id}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# -------------------------
# RAG CONTROL
# -------------------------
def get_rag_flag():
    if not os.path.exists(RAG_CONTROL_PATH):
        data = {"counter": 0}
    else:
        with open(RAG_CONTROL_PATH, "r") as f:
            data = json.load(f)

    data["counter"] += 1

    with open(RAG_CONTROL_PATH, "w") as f:
        json.dump(data, f, indent=4)

    return data["counter"] % 2 == 0


# -------------------------
# MEMORY
# ---------------------

def build_history_text(messages, last_k=6):
    history = ""

    for msg in messages[-last_k:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    return history


# -------------------------
# CONTEXT BUILDERS
# -------------------------
def build_context(results):
    context = ""

    for i, (_, row) in enumerate(results.iterrows()):
        context += (
            f"[ctx_{i}] Title: {row.get('title', '')}\n"
            f"Overview: {row.get('overview', '')}\n"
            f"Genres: {', '.join(row.get('genres', []))}\n"
            f"Keywords: {', '.join(row.get('keywords', []))}\n"
            f"---\n"
        )

    return context


# -------------------------
# PROMPT
# -------------------------
def build_prompt(query, context, history_text, session_id, user_language="es"):
    return f"""
SYSTEM ROLE:
You are a professional movie recommendation assistant with expertise in film genres, plots, and user preferences.

Your task is to provide concise, accurate, and context-grounded movie recommendations.

SESSION METADATA:
session_id: {session_id}
user_language: {user_language}

CONTEXT NOTE:
The retrieval system has already selected the most relevant context (either title-based or hybrid search).
You MUST ONLY use the provided context. Do NOT attempt to retrieve or infer external information.

CONVERSATION HISTORY:
{history_text}

USER QUERY:
{query}

RETRIEVAL CONTEXT:
{context}

LANGUAGE HANDLING:
1. The user may write in Spanish or English.
2. Always respond in Spanish if user_language == "es".
3. Do NOT translate movie titles; keep original titles exactly as in context.
4. If user explicitly asks in English, respond in English.

INSTRUCTIONS:
- Recommend between 3 and 5 movies maximum.
- Each recommendation must include:
  - Movie title
  - One short reason based strictly on the context
- Keep responses concise.
- Do NOT hallucinate information.
- If context is insufficient or irrelevant, provide a fallback.
- Always ground your reasoning in the context.

CONFIDENCE RULES:
- high → strong match with query and context
- medium → partial match
- low → weak or uncertain match

OUTPUT FORMAT (STRICT JSON):
Return ONLY valid JSON. No extra text.

{{
  "language": "es" or "en",
  "recommendations": [
    {{
      "title": "Movie Title",
      "reason": "Short explanation grounded in context",
      "confidence": "high|medium|low"
    }}
  ],
  "fallback": "One short clarification or suggestion in Spanish if needed"
}}

CONSTRAINTS:
- Maximum 2 short sentences per recommendation
- Do NOT include any text outside JSON
- Do NOT invent data not present in context

END
"""


# -------------------------
# RETRIEVAL LOGIC
# -------------------------
def retrieve(query, df, embeddings, faiss_index, use_rag):
    if use_rag:
        context, results = rag_retrieve(query, df, embeddings, faiss_index)
    else:
        results = hybrid_search_faiss(query, df, embeddings, faiss_index)
        context = build_context(results)

    return context, results


# -------------------------
# MAIN ORCHESTRATOR
# -------------------------
def run_agent(query, df, embeddings, faiss_index, session_id=None):

    if session_id is None:
        session_id = create_session()

    session = load_session(session_id)

    history_text = build_history_text(session["messages"])

    use_rag = get_rag_flag()

    context, results = retrieve(query, df, embeddings, faiss_index, use_rag)

    prompt = build_prompt(query, context, history_text, session_id)

    response = chat(
        model="qwen3:1.7b",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    # save interaction
    session["messages"].append({"role": "user", "content": query})
    session["messages"].append({"role": "assistant", "content": answer})

    session["rag_usage"].append({
        "query": query,
        "used_rag": use_rag
    })

    save_session(session_id, session)

    return {
        "response": answer,
        "session_id": session_id,
        "used_rag": use_rag
    }