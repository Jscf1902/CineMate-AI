import os
import pandas as pd

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
        embeddings, combined, index, _ = load_artifacts(path=EMB_PATH)
    else:
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


def get_orchestrator():
    # -------------------------
    # DATA
    # -------------------------
    df = pd.read_csv(DATA_PATH)

    # -------------------------
    # EMBEDDINGS + FAISS
    # -------------------------
    model, embeddings, index = load_or_create_embeddings(df)

    # -------------------------
    # COMPONENTES
    # -------------------------
    router_fn = route
    session_manager = SessionManager()
    prompt_builder = PromptBuilder()
    llm = generate_response

    retrieval = HybridSearch(
        faiss_index=index,
        metadata=df.to_dict(orient="records"),
        embeddings_model=model
    )

    # -------------------------
    # ORCHESTRATOR
    # -------------------------
    orchestrator = Orchestrator(
        router=router_fn,
        retrieval=retrieval,
        prompt_builder=prompt_builder,
        llm_client=llm,
        session_manager=session_manager,
        metadata=df.to_dict(orient="records")
    )

    return orchestrator