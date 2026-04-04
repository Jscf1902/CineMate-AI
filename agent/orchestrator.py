from typing import Optional
from src.rag.rag_pipeline import rag_retrieve
from src.retrieval.hybrid_search import hybrid_search

from agent.session_manager import (
    create_session,
    load_session,
    save_session,
    build_history
)

from agent.router import use_rag
from agent.prompt_builder import build_prompt
from agent.llm_client import generate_response


# retrieval
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


# main
def run_agent(
    query: str,
    df,
    embedding_service,
    embeddings,
    session_id: Optional[str] = None,
    user_language: str = "es"
):

    # session
    if session_id is None:
        session_id = create_session()

    session = load_session(session_id)

    # history
    history_text = build_history(session["messages"])

    # routing
    use_rag_flag = use_rag()

    # retrieval
    context, results = _retrieve(
        query=query,
        df=df,
        embedding_service=embedding_service,
        embeddings=embeddings,
        use_rag_flag=use_rag_flag
    )

    # prompt
    prompt = build_prompt(
        query=query,
        context=context,
        history_text=history_text,
        session_id=session_id,
        user_language=user_language
    )

    # llm
    answer = generate_response(prompt)

    # save
    session["messages"].append({"role": "user", "content": query})
    session["messages"].append({"role": "assistant", "content": answer})

    session["rag_usage"].append({
        "query": query,
        "used_rag": use_rag_flag
    })

    save_session(session_id, session)

    return {
        "response": answer,
        "session_id": session_id,
        "used_rag": use_rag_flag
    }