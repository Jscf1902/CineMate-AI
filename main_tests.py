import os
import warnings
import pandas as pd

# -------------------------
# silence HF logs
# -------------------------

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

# -------------------------
# imports
# -------------------------

from sentence_transformers import SentenceTransformer

from src.embeddings.embeddings_faiss import (
    generate_embeddings,
    build_faiss_index,
    save_artifacts,
    load_artifacts
)

from app.chat_cli import chat


# -------------------------
# config
# -------------------------

DATA_PATH = "data/raw/tmdb_movies_dataset.csv"
EMB_PATH = "data/processed"
MODEL_NAME = "all-MiniLM-L6-v2"


# -------------------------
# embedding service
# -------------------------

class EmbeddingService:
    def __init__(self, model, index):
        self.model = model
        self.index = index

    def encode_query(self, query: str):
        return self.model.encode(
            [query],
            normalize_embeddings=True
        )


# -------------------------
# load or create embeddings
# -------------------------

def load_or_create_embeddings(df):

    required_files = [
        "emb_title.npy",
        "emb_overview.npy",
        "emb_keywords.npy",
        "emb_genres.npy",
        "emb_combined.npy",
        "faiss.index"
    ]

    files_exist = all(
        os.path.exists(os.path.join(EMB_PATH, f))
        for f in required_files
    )

    model = SentenceTransformer(MODEL_NAME)

    if files_exist:
        print("embeddings listos")
        embeddings, combined, index, _ = load_artifacts(path=EMB_PATH)
    else:
        print("creando embeddings...")

        embeddings = generate_embeddings(df, model)
        combined, index = build_faiss_index(embeddings)

        save_artifacts(
            embeddings=embeddings,
            combined=combined,
            index=index,
            path=EMB_PATH,
            model_name=MODEL_NAME
        )

    service = EmbeddingService(model, index)

    return service, embeddings


# -------------------------
# main
# -------------------------

def main():

    df = pd.read_csv(DATA_PATH)

    service, embeddings = load_or_create_embeddings(df)

    chat(df, service, embeddings)


if __name__ == "__main__":
    main()