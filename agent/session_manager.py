import os
import json
import uuid
from datetime import datetime


BASE_PATH = "interactions"


# ruta de archivo
def _session_path(session_id: str) -> str:
    return os.path.join(BASE_PATH, f"session_{session_id}.json")


# crear sesión
def create_session() -> str:
    session_id = str(uuid.uuid4())

    data = {
        "session_id": session_id,
        "created_at": str(datetime.now()),
        "messages": [],
        "rag_usage": []
    }

    os.makedirs(BASE_PATH, exist_ok=True)

    with open(_session_path(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return session_id


# cargar sesión
def load_session(session_id: str) -> dict:
    with open(_session_path(session_id), "r", encoding="utf-8") as f:
        return json.load(f)


# guardar sesión
def save_session(session_id: str, data: dict) -> None:
    with open(_session_path(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# historial corto
def build_history(messages: list, last_k: int = 6) -> str:
    lines = []

    for msg in messages[-last_k:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)