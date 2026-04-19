import json
from agent.llm_client import generate_response


# -------------------------
# LLM intent detection
# -------------------------
def _classify_intent_llm(query: str):

    prompt = f"""
Classify the user intent.

Return ONLY one word from:
- vague
- more
- seen
- normal

Examples:
"si" → vague
"otra" → more
"ya la vi" → seen
"quiero algo de acción" → normal

User:
{query}
"""

    try:
        res = generate_response(prompt)
        label = res["content"].strip().lower()

        if label in ["vague", "more", "seen", "normal"]:
            return label

    except:
        pass

    return "normal"


# -------------------------
# fallback rules (rápidas)
# -------------------------
def _rule_based_intent(query: str):

    q = query.lower().strip()

    if q in ["si", "sí", "ok", "dale", "claro"]:
        return "vague"

    if any(x in q for x in ["otra", "dame otra"]):
        return "more"

    if any(x in q for x in ["ya la vi", "ya las vi"]):
        return "seen"

    return "normal"


# -------------------------
# intent wrapper
# -------------------------
def _get_intent(query: str):

    # primero reglas (rápido)
    intent = _rule_based_intent(query)

    if intent != "normal":
        return intent

    # si no está claro → LLM
    return _classify_intent_llm(query)


# -------------------------
# validar query previa
# -------------------------
def _is_valid_query(q: str):
    if not q:
        return False

    q = q.lower()

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

    intent = _get_intent(query)

    # -------------------------
    # vague → continuidad
    # -------------------------
    if intent == "vague":

        if _is_valid_query(last_query):
            return last_query

        pref_text = " ".join(
            preferences.get("genres", []) +
            preferences.get("keywords", [])
        )

        if pref_text:
            return pref_text

        return query

    # -------------------------
    # more → diversidad
    # -------------------------
    if intent == "more" and _is_valid_query(last_query):
        return last_query + " diferente"

    # -------------------------
    # seen → evitar repetición
    # -------------------------
    if intent == "seen" and _is_valid_query(last_query):
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