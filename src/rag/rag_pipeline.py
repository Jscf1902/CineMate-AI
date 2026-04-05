import pandas as pd
from difflib import SequenceMatcher


# -------------------------
# similitud
# -------------------------
def _similar(a: str, b: str):
    return SequenceMatcher(None, a, b).ratio()


# -------------------------
# detectar titulo (fuzzy)
# -------------------------
def _detect_title(query: str, df: pd.DataFrame, threshold=0.6):

    q = query.lower()

    best_score = 0
    best_idx = None

    for idx, title in enumerate(df["title"].astype(str)):
        t = title.lower()

        score = _similar(q, t)

        if score > best_score:
            best_score = score
            best_idx = idx

    if best_score >= threshold:
        return best_idx

    return None


# -------------------------
# construir query
# -------------------------
def _build_query_from_movie(row):

    keywords = row.get("keywords", [])
    genres = row.get("genres", [])

    if not isinstance(keywords, list):
        keywords = []

    if not isinstance(genres, list):
        genres = []

    kw_text = " ".join([str(x) for x in keywords])
    gen_text = " ".join([str(x) for x in genres])

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

        if not isinstance(genres, list):
            genres = []

        if not isinstance(keywords, list):
            keywords = []

        context += (
            f"Title: {title}\n"
            f"Overview: {overview}\n"
            f"Genres: {', '.join(map(str, genres))}\n"
            f"Keywords: {', '.join(map(str, keywords))}\n"
            f"---\n"
        )

    return context


# -------------------------
# fallback retrieval
# -------------------------
def _fallback_retrieval(query, df, embedding_service, embeddings, top_k):

    from src.retrieval.hybrid_search import hybrid_search

    # 1. intento normal
    results = hybrid_search(
        query=query,
        df=df,
        embedding_service=embedding_service,
        embeddings=embeddings,
        top_k=top_k
    )

    if results is not None and len(results) > 0:
        return results

    # 2. intento relajado (sin filtros implícitos)
    results = hybrid_search(
        query=" ".join(query.split()[:2]),  # query más simple
        df=df,
        embedding_service=embedding_service,
        embeddings=embeddings,
        top_k=top_k
    )

    if results is not None and len(results) > 0:
        return results

    # 3. fallback final → top global
    return df.sample(n=min(top_k, len(df)))


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

    idx = _detect_title(query, df)

    # -------------------------
    # caso con titulo
    # -------------------------
    if idx is not None:

        row = df.iloc[idx]

        new_query = _build_query_from_movie(row)

        results = hybrid_search(
            query=new_query,
            df=df,
            embedding_service=embedding_service,
            embeddings=embeddings,
            top_k=top_k + 1
        )

        results = results[results["title"] != row["title"]]

    # -------------------------
    # caso normal
    # -------------------------
    else:
        results = hybrid_search(
            query=query,
            df=df,
            embedding_service=embedding_service,
            embeddings=embeddings,
            top_k=top_k
        )

    # -------------------------
    # fallback robusto
    # -------------------------
    if results is None or len(results) == 0:
        results = _fallback_retrieval(
            query, df, embedding_service, embeddings, top_k
        )

    context = _build_context(results)

    return context, results.head(top_k)