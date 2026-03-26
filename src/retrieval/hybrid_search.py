import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def hybrid_search_faiss(
    query,
    df,
    embeddings,
    faiss_index,
    top_k=10,
    candidate_k=100,
    model_name="all-MiniLM-L6-v2",
):
    """
    Perform hybrid search using FAISS for candidate retrieval and
    dynamic multi-criteria scoring for final ranking.

    The scoring adapts based on the presence of keywords:
    - If keywords exist → higher weight on keywords
    - If keywords are missing → redistribute weight to title, genres, and overview

    Parameters
    ----------
    query : str
        User query text.
    df : pandas.DataFrame
        Dataset containing movie information.
    embeddings : dict
        Dictionary with embeddings per field.
    faiss_index : faiss.Index
        FAISS index built on overview embeddings.
    top_k : int, optional
        Number of final results to return.
    candidate_k : int, optional
        Number of candidates retrieved from FAISS.
    model_name : str, optional
        SentenceTransformer model name.

    Returns
    -------
    pandas.DataFrame
        Top-k results with associated scores.
    """

    model = SentenceTransformer(model_name)

    # Encode query
    query_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)

    # Retrieve candidates from FAISS
    _, indices = faiss_index.search(query_emb, candidate_k)
    candidate_indices = indices[0]

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    scores = []
    for i in candidate_indices:
        sim_title = cosine_sim(query_emb[0], embeddings["title"][i])
        sim_overview = cosine_sim(query_emb[0], embeddings["overview"][i])
        sim_keywords = cosine_sim(query_emb[0], embeddings["keywords"][i])
        sim_genres = cosine_sim(query_emb[0], embeddings["genres"][i])

        # Dynamic weighting based on keyword availability
        if not df.iloc[i]["keywords"]:
            weights = {
                "title": 0.45,
                "genres": 0.20,
                "overview": 0.35,
                "keywords": 0.0,
            }
        else:
            weights = {
                "title": 0.25,
                "genres": 0.10,
                "overview": 0.15,
                "keywords": 0.50,
            }

        score = (
            weights["title"] * sim_title
            + weights["overview"] * sim_overview
            + weights["keywords"] * sim_keywords
            + weights["genres"] * sim_genres
        )
        scores.append((i, score))

    # Rank results
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    final_indices = [i for i, _ in scores_sorted]
    final_scores = [s for _, s in scores_sorted]

    results = df.iloc[final_indices].copy()
    results["score"] = final_scores

    return results