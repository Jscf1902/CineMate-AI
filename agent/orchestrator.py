from src.rag.rag_pipeline import rag_retrieve
from src.retrieval.hybrid_search import hybrid_search

from agent.session_manager import (
    create_session,
    load_session,
    save_session,
    build_history
)

from agent.router import route
from agent.prompt_builder import build_prompt
from agent.llm_client import generate_response


# -------------------------
# extraer preferencias desde query
# -------------------------
def _extract_preferences(query: str):

    q = query.lower()

    genres = []
    themes = []

    GENRE_MAP = {
        "action": ["accion", "action", "peleas"],
        "horror": ["terror", "horror"],
        "comedy": ["comedia"],
        "drama": ["drama"],
        "romance": ["amor", "romance"],
        "sci-fi": ["ciencia ficcion", "futurista"],
        "fantasy": ["fantasia"],
        "anime": ["anime"],
    }

    THEME_MAP = {
        "robots": ["robots", "androides"],
        "zombies": ["zombies"],
        "space": ["espacio"],
        "war": ["guerra"],
        "magic": ["magia", "espadas"],
        "superheroes": ["superheroes"],
    }

    for g, words in GENRE_MAP.items():
        if any(w in q for w in words):
            genres.append(g)

    for t, words in THEME_MAP.items():
        if any(w in q for w in words):
            themes.append(t)

    return genres, themes


# -------------------------
# update memory
# -------------------------
def _update_memory(session: dict, query: str, results):

    session["memory"]["last_query"] = query

    # guardar últimas películas
    if results is not None and len(results) > 0:
        session["memory"]["last_movies"] = results["title"].tolist()

    # actualizar preferencias
    genres, themes = _extract_preferences(query)

    prev_genres = session["memory"]["preferences"].get("genres", [])
    prev_themes = session["memory"]["preferences"].get("keywords", [])

    session["memory"]["preferences"]["genres"] = list(set(prev_genres + genres))
    session["memory"]["preferences"]["keywords"] = list(set(prev_themes + themes))


# -------------------------
# retrieval
# -------------------------
def _retrieve(query, df, embedding_service, embeddings, use_rag_flag, session):

    exclude_titles = session["memory"].get("last_movies", [])

    if use_rag_flag:
        return rag_retrieve(
            query=query,
            df=df,
            embedding_service=embedding_service,
            embeddings=embeddings,
            top_k=5
        )

    results = hybrid_search(
        query=query,
        df=df,
        embedding_service=embedding_service,
        embeddings=embeddings,
        top_k=5,
        exclude_titles=exclude_titles
    )

    context = "\n".join(
        f"{str(row.get('title',''))} - {str(row.get('overview',''))}"
        for _, row in results.iterrows()
    )

    return context, results


# -------------------------
# main
# -------------------------
def run_agent(
    query: str,
    df,
    embedding_service,
    embeddings,
    session_id=None,
    user_language: str = "es"
):

    if session_id is None:
        session_id = create_session()

    session = load_session(session_id)
    if "rag_usage" not in session:
        session["rag_usage"] = []
    
    history_text = build_history(session["messages"])

    routing = route(query, session.get("memory", {}))

    enriched_query = routing["query"]
    use_rag_flag = session.get("use_rag", True)

    context, results = _retrieve(
        query=enriched_query,
        df=df,
        embedding_service=embedding_service,
        embeddings=embeddings,
        use_rag_flag=use_rag_flag,
        session=session
    )

    prompt = build_prompt(
        query=query,
        context=context,
        history_text=history_text,
        session_id=session_id,
        user_language=user_language
    )

    llm_result = generate_response(prompt)

    answer = llm_result["content"]
    latency = llm_result["latency_ms"]

    session["messages"].append({"role": "user", "content": query})
    session["messages"].append({"role": "assistant", "content": answer})

    session["rag_usage"].append({
        "query": query,
        "used_rag": use_rag_flag,
        "latency_ms": latency
    })

    _update_memory(session, enriched_query, results)

    save_session(session_id, session)

    return {
        "response": answer,
        "session_id": session_id,
        "used_rag": use_rag_flag,
        "latency_ms": latency
    }