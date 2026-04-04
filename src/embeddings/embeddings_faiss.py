import os
import json
import faiss
import numpy as np
import pandas as pd
from typing import Tuple
from sentence_transformers import SentenceTransformer


# genera embeddings por campo
def generate_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int = 64
) -> dict:

    def safe_join(x):
        return ", ".join(x) if isinstance(x, list) and len(x) > 0 else ""

    titles = df["title"].fillna("").astype(str).tolist()
    overviews = df["overview"].fillna("").astype(str).tolist()
    keywords = df["keywords"].apply(safe_join).tolist()
    genres = df["genres"].apply(safe_join).tolist()

    emb_title = model.encode(
        titles, batch_size=batch_size, normalize_embeddings=True
    )
    emb_overview = model.encode(
        overviews, batch_size=batch_size, normalize_embeddings=True
    )
    emb_keywords = model.encode(
        keywords, batch_size=batch_size, normalize_embeddings=True
    )
    emb_genres = model.encode(
        genres, batch_size=batch_size, normalize_embeddings=True
    )

    return {
        "title": np.array(emb_title, dtype="float32"),
        "overview": np.array(emb_overview, dtype="float32"),
        "keywords": np.array(emb_keywords, dtype="float32"),
        "genres": np.array(emb_genres, dtype="float32"),
    }


# mezcla embeddings y construye faiss
def build_faiss_index(
    embeddings: dict,
    weights: dict = None # type: ignore
) -> Tuple[np.ndarray, faiss.Index]:

    if weights is None:
        weights = {
            "title": 0.1,
            "overview": 0.25,
            "keywords": 0.6,
            "genres": 0.05,
        }

    emb_title = embeddings["title"]
    emb_overview = embeddings["overview"]
    emb_keywords = embeddings["keywords"]
    emb_genres = embeddings["genres"]

    combined = (
        weights["title"] * emb_title +
        weights["overview"] * emb_overview +
        weights["keywords"] * emb_keywords +
        weights["genres"] * emb_genres
    )

    faiss.normalize_L2(combined)

    dim = combined.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(combined) # type: ignore

    return combined, index


# guarda todo
def save_artifacts(
    embeddings: dict,
    combined: np.ndarray,
    index: faiss.Index,
    path: str = "data/processed",
    model_name: str = "all-MiniLM-L6-v2",
    weights: dict = None # type: ignore
):

    os.makedirs(path, exist_ok=True)

    np.save(f"{path}/emb_title.npy", embeddings["title"])
    np.save(f"{path}/emb_overview.npy", embeddings["overview"])
    np.save(f"{path}/emb_keywords.npy", embeddings["keywords"])
    np.save(f"{path}/emb_genres.npy", embeddings["genres"])
    np.save(f"{path}/emb_combined.npy", combined)

    faiss.write_index(index, f"{path}/faiss.index")

    metadata = {
        "model": model_name,
        "weights": weights,
        "normalized": True,
        "dim": int(combined.shape[1]),
        "size": int(combined.shape[0]),
    }

    with open(f"{path}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# carga todo
def load_artifacts(
    path: str = "data/processed"
) -> Tuple[dict, np.ndarray, faiss.Index, dict]:

    embeddings = {
        "title": np.load(f"{path}/emb_title.npy"),
        "overview": np.load(f"{path}/emb_overview.npy"),
        "keywords": np.load(f"{path}/emb_keywords.npy"),
        "genres": np.load(f"{path}/emb_genres.npy"),
    }

    combined = np.load(f"{path}/emb_combined.npy")
    index = faiss.read_index(f"{path}/faiss.index")

    metadata = {}
    meta_path = f"{path}/metadata.json"

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return embeddings, combined, index, metadata