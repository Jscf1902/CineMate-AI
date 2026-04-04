import sys

from sentence_transformers import SentenceTransformer

from agent.orchestrator import run_agent
from agent.session_manager import create_session
from src.embeddings.embeddings_faiss import load_artifacts


# carga inicial
def load_system():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings, combined, index, _ = load_artifacts()

    class EmbeddingService:
        def __init__(self, model, index):
            self.model = model
            self.index = index

        def encode_query(self, query: str):
            return self.model.encode(
                [query],
                normalize_embeddings=True
            )

    service = EmbeddingService(model, index)

    return service, embeddings


# loop principal
def chat():
    print("chat iniciado (escribe 'salir' para terminar)\n")

    service, embeddings = load_system()

    session_id = create_session()

    while True:
        query = input("usuario: ").strip()

        if query.lower() in ["salir", "exit", "quit"]:
            print("\nfin de la sesión")
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
            print(f"error: {e}")


# entrypoint
if __name__ == "__main__":
    # debes cargar tu dataset aquí
    import pandas as pd

    df = pd.read_csv("data/raw/tmdb_movies_dataset.csv")

    chat()