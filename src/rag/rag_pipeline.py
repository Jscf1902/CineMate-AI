import re
from typing import Tuple

import pandas as pd

from src.retrieval.hybrid_search import hybrid_search_faiss


# -------------------------
# TITLE SEARCH
# -------------------------
def search_by_title(query: str, df: pd.DataFrame, top_k: int = 5):
    """
    Search movies by title using simple string matching.
    """
    query = query.lower()

    matches = df[df["title"].str.lower().str.contains(query, na=False)]

    if matches.empty:
        return None

    return matches.head(top_k)


# -------------------------
# QUERY TYPE DETECTION
# -------------------------
def is_title_query(query: str) -> bool:
    """
    Detect if query is likely a title search.
    """
    query = query.strip()

    if len(query.split()) <= 3:
        return True

    if re.match(r"^[A-Z][a-z]+", query):
        return True

    return False


# -------------------------
# CONTEXT BUILDER
# -------------------------
def build_context(results: pd.DataFrame) -> str:
    """
    Convert retrieved results into a text context.
    """
    context = ""

    for _, row in results.iterrows():
        context += (
            f"Title: {row.get('title', '')}\n"
            f"Overview: {row.get('overview', '')}\n"
            f"Genres: {', '.join(row.get('genres', []))}\n"
            f"Keywords: {', '.join(row.get('keywords', []))}\n"
            f"---\n"
        )

    return context


# -------------------------
# HYBRID SEARCH WRAPPER
# -------------------------
def search_hybrid(query, df, embeddings, faiss_index, top_k=5):
    """
    Wrapper for hybrid search.
    """
    return hybrid_search_faiss(
        query=query,
        df=df,
        embeddings=embeddings,
        faiss_index=faiss_index,
        top_k=top_k,
    )


# -------------------------
# MAIN RAG RETRIEVAL
# -------------------------
def rag_retrieve(
    query: str,
    df: pd.DataFrame,
    embeddings: dict,
    faiss_index,
    top_k: int = 5
) -> Tuple[str, pd.DataFrame]:
    """
    Retrieve relevant documents for RAG.

    Returns
    -------
    context : str
    results : pandas.DataFrame
    """

    if is_title_query(query):
        results = search_by_title(query, df, top_k=top_k)

        if results is None:
            results = search_hybrid(query, df, embeddings, faiss_index, top_k=top_k)
    else:
        results = search_hybrid(query, df, embeddings, faiss_index, top_k=top_k)

    context = build_context(results)

    return context, results