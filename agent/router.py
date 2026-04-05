import os
import json


BASE_PATH = "interactions"
RAG_CONTROL_PATH = os.path.join(BASE_PATH, "rag_control.json")


def _use_rag() -> bool:
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


def _is_feedback(query: str) -> bool:
    q = query.lower()

    return any(x in q for x in [
        "ya la vi", "ya las vi", "otra", "dame otra",
        "no me gusta", "algo diferente"
    ])


def _enrich_query(query: str, memory: dict) -> str:
    last_query = memory.get("last_query", "")
    prefs = memory.get("preferences", {})

    # continuidad tipo "sí"
    if _is_vague(query) and last_query:
        return last_query

    # feedback tipo "otra"
    if _is_feedback(query) and last_query:
        return last_query + " diferente"

    return query


def route(query: str, memory: dict):
    enriched_query = _enrich_query(query, memory)

    return {
        "query": enriched_query,
        "use_rag": True
    }