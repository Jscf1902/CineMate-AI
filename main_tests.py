import os
import warnings
import pandas as pd

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer

from src.embeddings.embeddings_faiss import (
    generate_embeddings,
    build_faiss_index,
    save_artifacts,
    load_artifacts
)

from agent.orchestrator import Orchestrator
from agent.router import route
from agent.prompt_builder import PromptBuilder
from agent.session_manager import SessionManager
from src.retrieval.hybrid_search import HybridSearch
from agent.llm_client import generate_response


DATA_PATH = "data/raw/tmdb_movies_dataset.csv"
EMB_PATH = "data/processed"
MODEL_NAME = "all-MiniLM-L6-v2"


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

    return model, embeddings, index


def main():

    df = pd.read_csv(DATA_PATH)

    model, embeddings, index = load_or_create_embeddings(df)

    router = route
    session_manager = SessionManager()
    prompt_builder = PromptBuilder()
    llm = generate_response

    retrieval = HybridSearch(
        faiss_index=index,
        metadata=df.to_dict(orient="records"),
        embeddings_model=model
    )

    orchestrator = Orchestrator(
        router=router,
        retrieval=retrieval,
        prompt_builder=prompt_builder,
        llm_client=llm,
        session_manager=session_manager,
        metadata=df.to_dict(orient="records")
    )

    session_id = session_manager.create_session()

    mode = orchestrator.init_session_mode(session_id)

    print("\nSoy CineMate")
    print("Puedo recomendarte películas según lo que te guste.\n")
    print(f"[MODO SESIÓN: {mode}]\n")

    while True:
        user_input = input("usuario: ")

        if user_input.lower() in ["salir", "exit", "quit"]:
            print("\nfin")
            break

        print("\nCineMate está buscando recomendaciones...\n")

        response, latency, mode = orchestrator.handle_message(session_id, user_input)

        print("assistant:\n")
        print(response)

        print(f"\n(modo: {mode})")
        time = round(latency, 2)/60
        print(f"(tiempo: {time} min)\n")


if __name__ == "__main__":
    main()