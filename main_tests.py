import os
import warnings
import time
import pandas as pd

# -------------------------
# silence HF logs
# -------------------------

# os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# warnings.filterwarnings("ignore")

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

from agent.orchestrator import Orchestrator
from agent.router import route
from agent.prompt_builder import PromptBuilder
from agent.session_manager import SessionManager
from src.retrieval.hybrid_search import HybridSearch
from agent.llm_client import generate_response


# -------------------------
# config
# -------------------------

DATA_PATH = "data/raw/tmdb_movies_dataset.csv"
EMB_PATH = "data/processed"
MODEL_NAME = "all-MiniLM-L6-v2"


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

    return model, embeddings, index


# -------------------------
# MAIN
# -------------------------

def main():

    df = pd.read_csv(DATA_PATH)

    model, embeddings, index = load_or_create_embeddings(df)

    # =========================
    # INIT COMPONENTS
    # =========================

    router = route  # 🔥 FIX
    session_manager = SessionManager()
    prompt_builder = PromptBuilder()
    llm = generate_response  # 🔥 FIX

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

    # =========================
    # CHAT LOOP
    # =========================

    print("\nSoy CineMate 🎬")
    print("Puedo recomendarte películas según lo que te guste.\n")

    session_id = "default"

    while True:
        user_input = input("usuario: ")

        if user_input.lower() in ["salir", "exit", "quit"]:
            print("\nfin")
            break

        print("\nCineMate está buscando recomendaciones... 🔎\n")

        response, latency = orchestrator.handle_message(session_id, user_input)

        # detectar si usó cache o RAG
        session = session_manager.get_session(session_id)
        used_cache = session.get("current_index", 0) > 1

        mode = "CACHE" if used_cache else "RAG"

        print("assistant:\n")
        print(response)

        print(f"\n(modo: {mode})")
        print(f"(tiempo: {round(latency, 2)} s)\n")


if __name__ == "__main__":
    main()