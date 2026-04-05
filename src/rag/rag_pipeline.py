import re
import pandas as pd


# -------------------------
# detectar titulo
# -------------------------
def _detect_title(query: str, df: pd.DataFrame):

    q = query.lower()

    for idx, title in enumerate(df["title"].astype(str)):
        if title.lower() in q:
            return idx

    return None


# -------------------------
# construir query desde pelicula
# -------------------------
def _build_query_from_movie(row):

    keywords = row.get("keywords", [])
    genres = row.get("genres", [])

    kw_text = " ".join(keywords) if isinstance(keywords, list) else ""
    gen_text = " ".join(genres) if isinstance(genres, list) else ""

    return f"{kw_text} {gen_text}".strip()


# -------------------------
# contexto
# -------------------------
def _build_context(results: pd.DataFrame):

    context = ""

    for _, row in results.iterrows():
        title = str(row.get("title", ""))
        overview = str(row.get("overview", ""))

        genres = row.get("genres", [])
        keywords = row.get("keywords", [])

        genres = ", ".join(genres) if isinstance(genres, list) else ""
        keywords = ", ".join(keywords) if isinstance(keywords, list) else ""

        context += (
            f"Title: {title}\n"
            f"Overview: {overview}\n"
            f"Genres: {genres}\n"
            f"Keywords: {keywords}\n"
            f"---\n"
        )

    return context


# -------------------------
# main
# -------------------------
def rag_retrieve(
    query: str,
    df: pd.DataFrame,
    embedding_service,
    embeddings: dict,
    top_k: int = 5
):

    from src.retrieval.hybrid_search import hybrid_search

    # -------------------------
    # 1. detectar titulo
    # -------------------------
    idx = _detect_title(query, df)

    if idx is not None:
        row = df.iloc[idx]

        # construir nueva query desde pelicula
        new_query = _build_query_from_movie(row)

        results = hybrid_search(
            query=new_query,
            df=df,
            embedding_service=embedding_service,
            embeddings=embeddings,
            top_k=top_k + 1
        )

        # quitar pelicula original
        results = results[results["title"] != row["title"]]

    else:
        # flujo normal
        results = hybrid_search(
            query=query,
            df=df,
            embedding_service=embedding_service,
            embeddings=embeddings,
            top_k=top_k
        )

    context = _build_context(results)

    return context, results.head(top_k)