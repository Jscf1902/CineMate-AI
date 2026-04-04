import re
from typing import Tuple
import pandas as pd

from src.retrieval.hybrid_search import hybrid_search


# detecta si es búsqueda por título
def is_title_query(query: str) -> bool:
    q = query.strip()

    if len(q.split()) <= 3:
        return True

    if re.match(r"^[A-Z][a-z]+", q):
        return True

    return False


# búsqueda por título
def search_by_title(
    query: str,
    df: pd.DataFrame,
    top_k: int
):
    q = query.lower()

    results = df[df["title"].str.lower().str.contains(q, na=False)]

    if results.empty:
        return None

    return results.head(top_k)


# construye contexto
def build_context(results: pd.DataFrame) -> str:
    if results is None or results.empty:
        return ""

    lines = []

    for _, row in results.iterrows():
        genres = row.get("genres", [])
        keywords = row.get("keywords", [])

        genres_str = ", ".join(genres) if isinstance(genres, list) else ""
        keywords_str = ", ".join(keywords) if isinstance(keywords, list) else ""

        text = (
            f"Title: {row.get('title', '')}\n"
            f"Overview: {row.get('overview', '')}\n"
            f"Genres: {genres_str}\n"
            f"Keywords: {keywords_str}"
        )

        lines.append(text)

    return "\n---\n".join(lines)


# retrieval principal
def rag_retrieve(
    query: str,
    df: pd.DataFrame,
    embedding_service,
    embeddings: dict,
    top_k: int = 5
) -> Tuple[str, pd.DataFrame]:

    results = None

    # intento por título
    if is_title_query(query):
        results = search_by_title(query, df, top_k)

    # fallback a hybrid
    if results is None or results.empty:
        results = hybrid_search(
            query=query,
            df=df,
            embedding_service=embedding_service,
            embeddings=embeddings,
            top_k=top_k,
        )

    context = build_context(results)

    return context, results