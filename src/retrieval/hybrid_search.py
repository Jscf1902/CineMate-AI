import numpy as np


# -------------------------
# maps
# -------------------------
GENRE_MAP = {
    "action": ["accion", "action", "peleas"],
    "horror": ["terror", "horror", "miedo"],
    "comedy": ["comedia", "funny"],
    "drama": ["drama"],
    "romance": ["romance", "amor"],
    "sci-fi": ["ciencia ficcion", "sci-fi", "futurista"],
    "fantasy": ["fantasia", "fantasy"],
    "anime": ["anime"],
}

THEME_MAP = {
    "robots": ["robots", "androides"],
    "zombies": ["zombies", "muertos vivientes"],
    "space": ["espacio", "space"],
    "war": ["guerra", "war"],
    "magic": ["magia", "magico", "espadas"],
    "superheroes": ["superheroes", "heroes"],
}


# -------------------------
# detectar filtros
# -------------------------
def _detect_filters(query: str):

    q = query.lower()

    genres = []
    themes = []

    for g, words in GENRE_MAP.items():
        if any(w in q for w in words):
            genres.append(g)

    for t, words in THEME_MAP.items():
        if any(w in q for w in words):
            themes.append(t)

    return {
        "genres": genres,
        "themes": themes
    }


# -------------------------
# match filtros
# -------------------------
def _match_filters(row, filters):

    genres = row.get("genres", [])
    keywords = row.get("keywords", [])

    if not isinstance(genres, list):
        genres = []

    if not isinstance(keywords, list):
        keywords = []

    safe_text = " ".join([str(x) for x in genres + keywords]).lower()

    # si hay filtros, deben cumplirse al menos parcialmente
    if filters["genres"]:
        if not any(g in safe_text for g in filters["genres"]):
            return False

    if filters["themes"]:
        if not any(t in safe_text for t in filters["themes"]):
            return False

    return True


# -------------------------
# main
# -------------------------
def hybrid_search(
    query: str,
    df,
    embedding_service,
    embeddings: dict,
    top_k: int = 10,
    candidate_k: int = 100,
    exclude_titles=None
):

    if exclude_titles is None:
        exclude_titles = []

    query_emb = embedding_service.encode_query(query)[0]

    scores, indices = embedding_service.index.search(
        np.array([query_emb]).astype("float32"),
        min(candidate_k, len(df))
    )

    candidate_indices = indices[0]

    filters = _detect_filters(query)

    results = []

    for i in candidate_indices:
        if i == -1:
            continue

        row = df.iloc[i]
        title = row["title"]

        # excluir ya recomendadas
        if title in exclude_titles:
            continue

        # filtro por dominio
        if not _match_filters(row, filters):
            continue

        sim_title = np.dot(query_emb, embeddings["title"][i])
        sim_overview = np.dot(query_emb, embeddings["overview"][i])
        sim_keywords = np.dot(query_emb, embeddings["keywords"][i])
        sim_genres = np.dot(query_emb, embeddings["genres"][i])

        keywords = row.get("keywords", [])
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