import os
import warnings
import pandas as pd

from sentence_transformers import SentenceTransformer

from src.embeddings.embeddings_faiss import (
    generate_embeddings,
    build_faiss_index,
    save_artifacts,
    load_artifacts
)

from agent.orchestrator import run_agent
from agent.session_manager import create_session


# -------------------------
# config
# -------------------------

DATA_PATH = "data/raw/tmdb_movies_dataset.csv"
EMB_PATH = "data/processed"
MODEL_NAME = "all-MiniLM-L6-v2"


# -------------------------
# silence logs
# -------------------------

warnings.filterwarnings("ignore")


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
# check or build embeddings
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
        print("cargando embeddings...")
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
# chat loop
# -------------------------

def chat(df, service, embeddings):

    session_id = create_session()

    print("\nchat iniciado (salir para terminar)\n")

    while True:
        query = input("usuario: ").strip()

        if query.lower() in ["salir", "exit", "quit"]:
            print("\nfin\n")
            break

        try:
            result = run_agent(
                query=query,
                df=df,
                embedding_service=service,
                embeddings=embeddings,
                session_id=session_id
            )

            print("\nassistant:")
            print(result["response"])
            print("\n---\n")

        except Exception as e:
            print("error:", e)


# -------------------------
# main
# -------------------------

def main():

    df = pd.read_csv(DATA_PATH)

    service, embeddings = load_or_create_embeddings(df)

    chat(df, service, embeddings)


if __name__ == "__main__":
    main()