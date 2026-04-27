import os
import json
import faiss
import numpy as np
import pandas as pd
from typing import Tuple
from sentence_transformers import SentenceTransformer

# 1. GENERACIÓN: Ahora 4 veces más rápida al consolidar el paso por el modelo
def generate_embeddings(
    df: pd.DataFrame, 
    model: SentenceTransformer, 
    batch_size: int = 128 
) -> dict:

    def _prepare_text(series):
        # Manejo de NaNs y listas vectorizado
        return series.apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x) if pd.notnull(x) else "")

    n = len(df)
    fields = ["title", "overview", "keywords", "genres"]
    
    # Concatenamos todos los textos en una sola lista gigante
    # Esto permite que la GPU trabaje sin interrupciones
    all_texts = []
    for field in fields:
        all_texts.extend(_prepare_text(df[field]).tolist())

    # Un solo paso por el modelo
    all_embeddings = model.encode(
        all_texts, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    # Repartimos los resultados de vuelta a su campo original
    return {
        "title": all_embeddings[0:n],
        "overview": all_embeddings[n:2*n],
        "keywords": all_embeddings[2*n:3*n],
        "genres": all_embeddings[3*n:4*n],
    }

# 2. CONSTRUCCIÓN: Operaciones vectorizadas puras
def build_faiss_index(
    embeddings: dict, 
    weights: dict = None 
) -> Tuple[np.ndarray, faiss.Index]:

    if weights is None:
        weights = {"title": 0.1, "overview": 0.25, "keywords": 0.6, "genres": 0.05}

    # Calculamos el combinado de una sola vez
    combined = sum(embeddings[k] * weights[k] for k in weights).astype("float32")
    
    # Normalización para que IndexFlatIP funcione como Similitud Coseno
    faiss.normalize_L2(combined)

    dim = combined.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(combined)

    return combined, index

# 3. GUARDADO: Mantiene los archivos .npy individuales para compatibilidad total
def save_artifacts(
    embeddings: dict, 
    combined: np.ndarray, 
    index: faiss.Index, 
    path: str = "data/processed", 
    model_name: str = "all-MiniLM-L6-v2", 
    weights: dict = None
):
    os.makedirs(path, exist_ok=True)

    # Guardado tradicional campo por campo
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

# 4. CARGA: Sin cambios, funciona con tus archivos actuales
def load_artifacts(path: str = "data/processed") -> Tuple[dict, np.ndarray, faiss.Index, dict]:
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