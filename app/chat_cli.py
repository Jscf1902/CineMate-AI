import json

from agent.orchestrator import run_agent
from agent.session_manager import create_session, load_session, save_session
from agent.llm_client import generate_response


# -------------------------
# helpers LLM ligeros
# -------------------------

def extract_name(text: str):
    prompt = f"""
Extrae el nombre de la siguiente frase.
Si no hay nombre responde null.

Input: "{text}"

Output (solo nombre o null):
"""
    
    result = generate_response(prompt)
    name = result["content"].strip().replace('"', '').replace("'", "")

    if name.lower() == "null" or len(name.split()) > 2:
        return None

    return name


def classify_intent(text: str):
    prompt = f"""
Clasifica la intención del usuario en una sola palabra:

- movie → si habla de películas
- personal → si habla de sí mismo
- other → cualquier otro caso

Input: "{text}"

Output:
"""

    result = generate_response(prompt)
    intent = result["content"].strip().lower()

    if "movie" in intent:
        return "movie"
    if "personal" in intent:
        return "personal"
    return "other"


# -------------------------
# chat
# -------------------------

def chat(df, service, embeddings):

    print("\nSoy CineMate 🎬")
    print("Puedo recomendarte películas según lo que te guste.\n")

    from agent.session_manager import create_session
    from agent.orchestrator import run_agent

    session_id = create_session()
    session = load_session(session_id)

    mode = "RAG activado" if session["use_rag"] else "Búsqueda directa"

    print(f"\nModo: {mode}\n")
    
    while True:
        query = input("usuario: ").strip()

        if query.lower() in ["salir", "exit", "quit"]:
            print("\nfin\n")
            break

        try:
            # mensaje UX
            print("\nCineMate está buscando recomendaciones... 🔎\n")

            result = run_agent(
                query=query,
                df=df,
                embedding_service=service,
                embeddings=embeddings,
                session_id=session_id
            )

            print("assistant:")
            print(result["response"])
            lat_min = result['latency_ms']/60000
            print(f"(tiempo: {lat_min:.2f} m)")
            print("\n---\n")

        except Exception as e:
            print("error:", e)