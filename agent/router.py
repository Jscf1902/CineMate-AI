import os
import json


BASE_PATH = "interactions"
RAG_CONTROL_PATH = os.path.join(BASE_PATH, "rag_control.json")

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


def _is_feedback_more(query: str) -> bool:
    q = query.lower()

    return any(x in q for x in [
        "otra", "dame otra", "algo diferente", "otra recomendacion"
    ])


def _is_feedback_seen(query: str) -> bool:
    q = query.lower()

    return any(x in q for x in [
        "ya la vi", "ya las vi", "ya me la vi", "ya me las vi"
    ])


def _enrich_query(query: str, memory: dict) -> str:

    last_query = memory.get("last_query", "")

    # continuidad tipo "sí"
    if _is_vague(query) and last_query:
        return last_query

    # pedir otra recomendación
    if _is_feedback_more(query) and last_query:
        return last_query + " diferente"

    # ya vistas → forzar diversidad
    if _is_feedback_seen(query) and last_query:
        return last_query + " diferente no repetir"

    return query


def route(query: str, memory: dict):

    enriched_query = _enrich_query(query, memory)

    return {
        "query": enriched_query,
        "use_rag": True
    }