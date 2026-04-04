import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_field_embeddings(df, model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings for selected fields in the dataset.
    Fields:
    - title
    - overview
    - keywords (joined)
    - genres (joined)
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing text fields.
    model_name : str, optional
        SentenceTransformer model name.
    Returns
    -------
    dict
        Dictionary containing numpy arrays of embeddings per field.
    """

    model = SentenceTransformer(model_name)
    titles = df["title"].fillna("").astype(str).tolist()
    overviews = df["overview"].fillna("").astype(str).tolist()
    keywords = df["keywords"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else ""
    ).tolist()
    genres = df["genres"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else ""
    ).tolist()
    emb_title = model.encode(titles, batch_size=64, show_progress_bar=False)
    emb_overview = model.encode(overviews, batch_size=64, show_progress_bar=False)
    emb_keywords = model.encode(keywords, batch_size=64, show_progress_bar=False)
    emb_genres = model.encode(genres, batch_size=64, show_progress_bar=False)
    return {
        "title": np.array(emb_title),
        "overview": np.array(emb_overview),
        "keywords": np.array(emb_keywords),
        "genres": np.array(emb_genres),
    }

def build_faiss_index(embeddings):
    """
    Build a FAISS index using overview embeddings.
    Parameters
    ----------
    embeddings : dict
        Dictionary containing embedding arrays.
    Returns
    -------
    faiss.Index
        FAISS index with normalized vectors.
    """

    emb = embeddings["overview"].astype("float32")
    dimension = emb.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(emb)
    index.add(emb) # type: ignore
    return index

def save_embeddings_and_index(embeddings, faiss_index, path="data/processed"):
    """
    Save embeddings and FAISS index to disk.
    """
    os.makedirs(path, exist_ok=True)

    np.save(f"{path}/emb_title.npy", embeddings["title"])
    np.save(f"{path}/emb_overview.npy", embeddings["overview"])
    np.save(f"{path}/emb_keywords.npy", embeddings["keywords"])
    np.save(f"{path}/emb_genres.npy", embeddings["genres"])

    faiss.write_index(faiss_index, f"{path}/faiss.index")
    
def load_embeddings_and_index(path="data/processed"):
    """
    Load embeddings and FAISS index from disk.
    """

    embeddings = {
        "title": np.load(f"{path}/emb_title.npy"),
        "overview": np.load(f"{path}/emb_overview.npy"),
        "keywords": np.load(f"{path}/emb_keywords.npy"),
        "genres": np.load(f"{path}/emb_genres.npy"),
    }

    faiss_index = faiss.read_index(f"{path}/faiss.index")

    return embeddings, faiss_index