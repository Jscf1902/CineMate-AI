import time
import hashlib
import os
import json

from src.retrieval.query_understanding import QueryUnderstanding


class Orchestrator:
    def __init__(self, router, retrieval, prompt_builder, llm_client, session_manager, metadata):
        self.router = router
        self.retrieval = retrieval
        self.prompt_builder = prompt_builder
        self.llm = llm_client
        self.session_manager = session_manager
        self.metadata = metadata

        self.query_analyzer = QueryUnderstanding(metadata)
        self.rag_path = "interactions/rag_counter.json"

    # =====================================================
    # GLOBAL RAG COUNTER
    # =====================================================
    def _get_rag_mode(self):
        if not os.path.exists(self.rag_path):
            data = {"counter": 0}
        else:
            with open(self.rag_path, "r") as f:
                data = json.load(f)

        counter = data.get("counter", 0)
        use_rag = counter % 2 == 0

        data["counter"] = counter + 1

        os.makedirs(os.path.dirname(self.rag_path), exist_ok=True)
        with open(self.rag_path, "w") as f:
            json.dump(data, f, indent=2)

        return use_rag

    # =====================================================
    # INIT MODE
    # =====================================================
    def init_session_mode(self, session_id):
        session = self.session_manager.get_session(session_id)

        if "mode" not in session or session["mode"] is None:
            use_rag = self._get_rag_mode()

            session["mode"] = "RAG" if use_rag else "DIRECT"
            session["use_rag"] = use_rag

            self.session_manager.save_session(session)

        return session["mode"]

    # =====================================================
    # MAIN
    # =====================================================
    def handle_message(self, session_id, user_input):
        start_time = time.time()

        session = self.session_manager.get_session(session_id)
        mode = session.get("mode", "RAG")

        # -------------------------
        # ROUTER
        # -------------------------
        route_result = self.router(user_input, session.get("memory", {}))
        routed_query = route_result["query"]

        # -------------------------
        # QUERY UNDERSTANDING
        # -------------------------
        analyzed = self.query_analyzer.analyze(routed_query, session.get("memory"))

        used_cache = False

        # -------------------------
        # DIRECT MODE
        # -------------------------
        if mode == "DIRECT":
            candidates = self.retrieval.search(
                analyzed_query=analyzed,
                memory=session.get("memory"),
                top_k=5
            )
            selected = candidates[:2]

        # -------------------------
        # RAG MODE
        # -------------------------
        else:
            use_cache = self._should_use_cache(analyzed, session)

            if use_cache:
                used_cache = True
                candidates = session.get("candidates", [])
                idx = session.get("current_index", 0)

                if idx < len(candidates):
                    selected = [candidates[idx]]
                    session["current_index"] += 1
                else:
                    selected = []
            else:
                candidates = self.retrieval.search(
                    analyzed_query=analyzed,
                    memory=session.get("memory"),
                    top_k=10
                )

                session["candidates"] = candidates
                session["current_index"] = 1
                session["last_query_signature"] = self._build_query_signature(analyzed)

                selected = candidates[:2]

        # -------------------------
        # EXTRAER DATOS PARA MÉTRICAS
        # -------------------------
        scores = [item.get("score", 0) for item in selected]
        titles = [item.get("title", "") for item in selected]

        # -------------------------
        # UPDATE MEMORY
        # -------------------------
        self._update_memory(session, selected)

        # -------------------------
        # PROMPT
        # -------------------------
        prompt = self.prompt_builder.build(
            user_input=user_input,
            context=selected,
            history=session.get("messages", [])
        )

        # -------------------------
        # LLM
        # -------------------------
        result = self.llm(prompt)
        response = result["content"]
        latency = result["latency_ms"] / 1000

        # -------------------------
        # TRACK INTERACTION (CLAVE)
        # -------------------------
        self.session_manager.track_interaction(session, {
            "query": user_input,
            "routed_query": routed_query,
            "mode": mode,
            "use_rag": True if mode == "RAG" else False,
            "used_cache": used_cache,
            "latency": latency,
            "scores": scores,
            "titles": titles,
            "results_count": len(selected)
        })

        # -------------------------
        # SAVE MESSAGES
        # -------------------------
        self.session_manager.add_message(session_id, "user", user_input)
        self.session_manager.add_message(session_id, "assistant", response)

        self.session_manager.save_session(session)

        return response, latency, mode

    # =====================================================
    # CACHE LOGIC
    # =====================================================
    def _should_use_cache(self, analyzed, session):
        if not session.get("candidates"):
            return False

        if analyzed["is_followup"]:
            return True

        last_signature = session.get("last_query_signature")
        current_signature = self._build_query_signature(analyzed)

        return last_signature == current_signature

    # =====================================================
    # SIGNATURE
    # =====================================================
    def _build_query_signature(self, analyzed):
        base = (
            (analyzed.get("title") or "") +
            " ".join(analyzed.get("genres", [])) +
            " ".join(analyzed.get("keywords", []))
        )
        return hashlib.md5(base.encode()).hexdigest()

    # =====================================================
    # MEMORY
    # =====================================================
    def _update_memory(self, session, selected):
        if "memory" not in session:
            session["memory"] = {}

        memory = session["memory"]

        if "last_movies" not in memory:
            memory["last_movies"] = []

        for item in selected:
            title = item.get("title")
            if title and title not in memory["last_movies"]:
                memory["last_movies"].append(title)