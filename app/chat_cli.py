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

    session_id = create_session()
    session = load_session(session_id)

    while True:
        query = input("usuario: ").strip()

        if query.lower() in ["salir", "exit", "quit"]:
            print("\nfin\n")
            break

        session = load_session(session_id)

        # -------------------------
        # detectar nombre
        # -------------------------
        if not session.get("user_name"):
            name = extract_name(query)

            if name:
                session["user_name"] = name
                save_session(session_id, session)

                print(f"\nEncantado {name} 👋")
                print("¿Qué tipo de película te gustaría ver?\n")
                continue

        # -------------------------
        # clasificar intención
        # -------------------------
        intent = classify_intent(query)

        # -------------------------
        # respuestas directas
        # -------------------------
        if intent == "personal":
            if "nombre" in query.lower():
                name = session.get("user_name", "no lo sé")
                print(f"\nTe llamas {name} 😉\n")
                continue

        if intent != "movie":
            print("\nPuedo ayudarte a encontrar películas 🎬")
            print("¿Buscas acción, terror o ciencia ficción?\n")
            continue

        # -------------------------
        # flujo normal (agente)
        # -------------------------
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