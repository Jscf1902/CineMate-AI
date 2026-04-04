import os
import json


BASE_PATH = "interactions"
RAG_CONTROL_PATH = os.path.join(BASE_PATH, "rag_control.json")


# alterna uso de rag
def use_rag() -> bool:
    if not os.path.exists(RAG_CONTROL_PATH):
        data = {"counter": 0}
    else:
        with open(RAG_CONTROL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

    data["counter"] += 1

    os.makedirs(BASE_PATH, exist_ok=True)

    with open(RAG_CONTROL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data["counter"] % 2 == 0