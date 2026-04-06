import json
import os
from datetime import datetime


class SessionManager:
    def __init__(self, storage_path="interactions"):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

    # =====================================================
    # PUBLIC
    # =====================================================
    def get_session(self, session_id):
        path = self._get_path(session_id)

        if not os.path.exists(path):
            session = self._init_session(session_id)
            self._save(session)
            return session

        with open(path, "r", encoding="utf-8") as f:
            session = json.load(f)

        return self._ensure_structure(session)

    def add_message(self, session_id, role, content):
        session = self.get_session(session_id)

        session["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        self._save(session)

    def save_session(self, session):
        self._save(session)

    # =====================================================
    # CANDIDATE CACHE
    # =====================================================
    def save_candidates(self, session, candidates, signature):
        session["candidates"] = candidates
        session["current_index"] = 0
        session["last_query_signature"] = signature

        self._save(session)

    def get_next_candidate(self, session):
        idx = session.get("current_index", 0)
        candidates = session.get("candidates", [])

        if idx >= len(candidates):
            return None

        session["current_index"] += 1
        self._save(session)

        return candidates[idx]

    def reset_candidates(self, session):
        session["candidates"] = []
        session["current_index"] = 0
        session["last_query_signature"] = ""
        self._save(session)

    # =====================================================
    # MEMORY MANAGEMENT
    # =====================================================
    def update_memory(self, session, selected_items):
        memory = session["memory"]

        for item in selected_items:
            title = item.get("title")

            if title and title not in memory["last_movies"]:
                memory["last_movies"].append(title)

            # actualizar preferencias
            for g in item.get("genres", []):
                if g not in memory["preferences"]["genres"]:
                    memory["preferences"]["genres"].append(g)

            for k in item.get("keywords", []):
                if k not in memory["preferences"]["keywords"]:
                    memory["preferences"]["keywords"].append(k)

        self._save(session)

    # =====================================================
    # INTERNAL
    # =====================================================
    def _init_session(self, session_id):
        return {
            "session_id": session_id,
            "messages": [],
            "memory": {
                "last_movies": [],
                "preferences": {
                    "genres": [],
                    "keywords": []
                }
            },
            "candidates": [],
            "current_index": 0,
            "last_query_signature": ""
        }

    def _ensure_structure(self, session):
        if "memory" not in session:
            session["memory"] = {}

        if "last_movies" not in session["memory"]:
            session["memory"]["last_movies"] = []

        if "preferences" not in session["memory"]:
            session["memory"]["preferences"] = {
                "genres": [],
                "keywords": []
            }

        if "candidates" not in session:
            session["candidates"] = []

        if "current_index" not in session:
            session["current_index"] = 0

        if "last_query_signature" not in session:
            session["last_query_signature"] = ""

        return session

    def _get_path(self, session_id):
        return os.path.join(self.storage_path, f"session_{session_id}.json")

    def _save(self, session):
        path = self._get_path(session["session_id"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)