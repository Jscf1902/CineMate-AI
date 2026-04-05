import numpy as np


def _is_same_title(query: str, title: str) -> bool:
    q = query.lower()
    t = str(title).lower()

    return t in q or q in t


def hybrid_search(
    query: str,
    df,
    embedding_service,
    embeddings: dict,
    top_k: int = 10,
    candidate_k: int = 100,
):
    """
    faiss + rerank + evita misma pelicula
    """

    query_emb = embedding_service.encode_query(query)[0]

    scores, indices = embedding_service.index.search(
        np.array([query_emb]).astype("float32"),
        min(candidate_k, len(df))
    )

    candidate_indices = indices[0]

    results = []

    for i in candidate_indices:
        if i == -1:
            continue

        title = df.iloc[i]["title"]

        # evitar misma pelicula
        if _is_same_title(query, title):
            continue

        sim_title = np.dot(query_emb, embeddings["title"][i])
        sim_overview = np.dot(query_emb, embeddings["overview"][i])
        sim_keywords = np.dot(query_emb, embeddings["keywords"][i])
        sim_genres = np.dot(query_emb, embeddings["genres"][i])

        keywords = df.iloc[i]["keywords"]
        has_keywords = isinstance(keywords, list) and len(keywords) > 0

        if has_keywords:
            w_title = 0.25
            w_overview = 0.10
            w_keywords = 0.60
            w_genres = 0.05
        else:
            w_title = 0.625
            w_overview = 0.25
            w_keywords = 0.0
            w_genres = 0.125

        score = (
            w_title * sim_title +
            w_overview * sim_overview +
            w_keywords * sim_keywords +
            w_genres * sim_genres
        )

        results.append((i, float(score)))

    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    final_indices = [i for i, _ in results]
    final_scores = [s for _, s in results]

    output = df.iloc[final_indices].copy()
    output["score"] = final_scores

    return output