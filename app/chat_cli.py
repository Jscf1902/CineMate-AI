import os
import pandas as pd

from sentence_transformers import SentenceTransformer

from agent.orchestrator import run_agent
from agent.session_manager import create_session, load_session, save_session
from src.embeddings.embeddings_faiss import load_artifacts


# carga sistema
def load_system():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings, combined, index, _ = load_artifacts(path="data/processed")

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


# chat
def chat(df):

    print("\nSoy CineMate 🎬")
    print("Puedo recomendarte películas según lo que te guste.")
    print("Puedes pedirme cosas como:")
    print("- 'películas de acción con robots'")
    print("- 'algo parecido a Inception'\n")

    name = input("¿Cómo te llamas?: ").strip()

    session_id = create_session()
    session = load_session(session_id)

    session["user_name"] = name
    save_session(session_id, session)

    print(f"\nEncantado {name}, ¿qué te gustaría ver?\n")

    service, embeddings = load_system()

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
            print(f"(tiempo: {result['latency_ms']} ms)")
            print("\n---\n")

        except Exception as e:
            print("error:", e)


# main
if __name__ == "__main__":
    df = pd.read_csv("data/raw/tmdb_movies_dataset.csv")
    chat(df)