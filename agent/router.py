import os
import json


BASE_PATH = "interactions"
RAG_CONTROL_PATH = os.path.join(BASE_PATH, "rag_control.json")


# alterna rag
def use_rag() -> bool:
    if not os.path.exists(RAG_CONTROL_PATH):
        data = {"counter": 0}
    else:
        with open(RAG_CONTROL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

    data["counter"] += 1

    os.makedirs(BASE_PATH, exist_ok=True)

    with open(RAG_CONTROL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data["counter"] % 2 == 0


# detecta query vaga
def _is_vague(query: str) -> bool:
    q = query.lower().strip()

    vague_terms = {
        "si", "sí", "ok", "dale", "me gusta",
        "algo así", "asi", "perfecto", "claro"
    }

    if q in vague_terms:
        return True

    if len(q.split()) <= 2:
        return True

    return False


# reconstruye query
def _enrich_query(query: str, memory: dict) -> str:
    if not memory:
        return query

    last_query = memory.get("last_query", "")

    if _is_vague(query) and last_query:
        return last_query

    return query


# router principal
def route(query: str, memory: dict):
    enriched_query = _enrich_query(query, memory)

    return {
        "query": enriched_query,
        "use_rag": use_rag()
    }