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

    return q in vague_terms or len(q.split()) <= 2


def _is_feedback_more(query: str) -> bool:
    q = query.lower()
    return any(x in q for x in [
        "otra", "dame otra", "algo diferente"
    ])


def _is_feedback_seen(query: str) -> bool:
    q = query.lower()
    return any(x in q for x in [
        "ya la vi", "ya las vi", "ya me la vi", "ya me las vi"
    ])


# -------------------------
# validar query previa
# -------------------------
def _is_valid_query(q: str) -> bool:
    if not q:
        return False

    q = q.lower()

    # evitar queries genéricas o fallback
    bad_patterns = [
        "peliculas",
        "algo",
        "recomendacion",
        "diferente"
    ]

    if any(p in q for p in bad_patterns) and len(q.split()) <= 3:
        return False

    return True


# -------------------------
# enriquecer query
# -------------------------
def _enrich_query(query: str, memory: dict):

    last_query = memory.get("last_query", "")
    preferences = memory.get("preferences", {})

    # continuidad tipo "sí"
    if _is_vague(query):
        if _is_valid_query(last_query):
            return last_query

        # fallback → usar preferencias
        pref_text = " ".join(
            preferences.get("genres", []) +
            preferences.get("keywords", [])
        )

        if pref_text:
            return pref_text

        return query

    # pedir otra recomendación
    if _is_feedback_more(query) and _is_valid_query(last_query):
        return last_query + " diferente"

    # ya vistas
    if _is_feedback_seen(query) and _is_valid_query(last_query):
        return last_query + " diferente no repetir"

    return query


# -------------------------
# router principal
# -------------------------
def route(query: str, memory: dict):

    enriched_query = _enrich_query(query, memory)

    return {
        "query": enriched_query,
        "use_rag": True
    }