import os
import logging
import warnings
from src.data.load_data import load_and_prepare_dataset
from src.retrieval.hybrid_search import hybrid_search_faiss
from src.embeddings.embeddings_faiss import (
    generate_field_embeddings,
    build_faiss_index,
    save_embeddings_and_index,
    load_embeddings_and_index,
)

# Silence transformers logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Silence warnings
warnings.filterwarnings("ignore")

# Disable tqdm globally
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_PATH = r"C:\Users\juans\OneDrive\Documentos\Maestria en Ingenieria y Analitica de Datos\Proyecto de Grado\CineMate AI\data\raw\tmdb_movies_dataset.csv"
EMB_PATH = "data/processed"

def load_or_create_embeddings(df):
    """
    Load embeddings and FAISS index if they exist,
    otherwise generate and persist them.
    """

    faiss_file = os.path.join(EMB_PATH, "faiss.index")

    if os.path.exists(faiss_file):
        print("\nLoading existing embeddings and FAISS index...")
        embeddings, faiss_index = load_embeddings_and_index(EMB_PATH)
    else:
        print("\nGenerating embeddings...")
        embeddings = generate_field_embeddings(df)

        print("\nBuilding FAISS index...")
        faiss_index = build_faiss_index(embeddings)

        print("\nSaving embeddings and index...")
        save_embeddings_and_index(embeddings, faiss_index, EMB_PATH)

    return embeddings, faiss_index


def run_tests(df, embeddings, faiss_index):
    """
    Execute a set of predefined queries to validate the retrieval system.
    """

    queries = [
        # General recommendations
        "space exploration and survival",
        "romantic movie with a sad ending",
        "action movie with a strong female lead",
        "movies about artificial intelligence",
        "movies about family and relationships",

        # Keyword-driven queries
        "space war alien future",
        "time travel paradox",
        "zombie apocalypse survival",
        "serial killer investigation",
        "magic fantasy kingdom",

        # Genre-based queries
        "science fiction adventure",
        "comedy drama",
        "horror thriller",
        "animated family movie",
        "historical war movie",

        # Title search
        "interstellar",
        "batman",
        "harry potter",
        "avengers",
        "joker",

        # Franchise / sequels
        "fast and furious",
        "mission impossible",
        "john wick",
        "transformers",
        "spiderman",

        # Ambiguous / semantic tests
        "avatar",
        "avatar blue people",
        "avatar animated",
        "king kong",
        "godzilla",
    ]

    for i, query in enumerate(queries, 1):
        print("\n" + "=" * 60)
        print(f"TEST {i}: {query}")
        print("=" * 60)

        results = hybrid_search_faiss(
            query=query,
            df=df,
            embeddings=embeddings,
            faiss_index=faiss_index,
            top_k=5,
        )

        for _, row in results.iterrows():
            print(f"- {row['title']} (Score: {round(row['score'], 4)})")


def main():
    print("\nLoading dataset...")
    df = load_and_prepare_dataset(DATA_PATH)

    embeddings, faiss_index = load_or_create_embeddings(df)

    print("\nRunning test queries...")
    run_tests(df, embeddings, faiss_index)

    print("\nProcess completed successfully.")


if __name__ == "__main__":
    main()