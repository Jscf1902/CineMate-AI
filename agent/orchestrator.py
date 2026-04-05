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


def _retrieve(query, df, embedding_service, embeddings, use_rag_flag, top_k=5):
    if use_rag_flag:
        return rag_retrieve(
            query=query,
            df=df,
            embedding_service=embedding_service,
            embeddings=embeddings,
            top_k=top_k
        )

    results = hybrid_search(
        query=query,
        df=df,
        embedding_service=embedding_service,
        embeddings=embeddings,
        top_k=top_k
    )

    context = "\n".join(
        f"{row['title']} - {row['overview']}"
        for _, row in results.iterrows()
    )

    return context, results


def _update_memory(session: dict, query: str, results):
    session["memory"]["last_query"] = query

    # guardar últimas películas recomendadas
    if results is not None and len(results) > 0:
        session["memory"]["last_movies"] = results["title"].tolist()


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

    history_text = build_history(session["messages"])
    user_name = session.get("user_name")

    routing = route(query, session.get("memory", {}))

    enriched_query = routing["query"]
    use_rag_flag = routing["use_rag"]

    context, results = _retrieve(
        query=enriched_query,
        df=df,
        embedding_service=embedding_service,
        embeddings=embeddings,
        use_rag_flag=use_rag_flag
    )

    prompt = build_prompt(
        query=query,
        context=context,
        history_text=history_text,
        session_id=session_id,
        user_language=user_language,
        user_name=user_name
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