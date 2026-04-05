import os
import json
import uuid
from datetime import datetime


BASE_PATH = "interactions"
COUNTER_PATH = os.path.join(BASE_PATH, "rag_counter.json")

def _session_path(session_id: str) -> str:
    return os.path.join(BASE_PATH, f"session_{session_id}.json")


def create_session() -> str:
    session_id = str(uuid.uuid4())

    data = {
        "session_id": session_id,
        "created_at": str(datetime.now()),
        "messages": [],
        "use_rag": _get_rag_mode(),
        "user_name": None,
        "memory": {
            "last_query": "",
            "last_intent": "",
            "last_movies": [],
            "preferences": {
                "genres": [],
                "keywords": []
            },
            "conversation_state": {
                "fallback_count": 0
            }
        }
    }

    os.makedirs(BASE_PATH, exist_ok=True)

    with open(_session_path(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return session_id


def load_session(session_id: str) -> dict:
    with open(_session_path(session_id), "r", encoding="utf-8") as f:
        return json.load(f)


def save_session(session_id: str, data: dict) -> None:
    with open(_session_path(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_history(messages: list, last_k: int = 6) -> str:
    lines = []

    for msg in messages[-last_k:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)

def _get_rag_mode():

    if not os.path.exists(COUNTER_PATH):
        data = {"counter": 0}
    else:
        with open(COUNTER_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

    counter = data["counter"]

    # alternar
    use_rag = counter % 2 == 0

    # actualizar
    data["counter"] += 1

    with open(COUNTER_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return use_rag