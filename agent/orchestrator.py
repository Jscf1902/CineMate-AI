import time
import hashlib

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

    # =====================================================
    # MAIN ENTRY
    # =====================================================
    def handle_message(self, session_id, user_input):
        start_time = time.time()

        session = self.session_manager.get_session(session_id)

        # 1. ROUTER
        routed_query = self.router.process(user_input, session)

        # 2. ANALYZE QUERY
        analyzed = self.query_analyzer.analyze(routed_query, session.get("memory"))

        # 3. DECIDE CACHE OR NEW SEARCH
        use_cache = self._should_use_cache(analyzed, session)

        if use_cache:
            candidates = session.get("candidates", [])
            current_index = session.get("current_index", 0)

            if current_index < len(candidates):
                selected = [candidates[current_index]]
                session["current_index"] += 1
            else:
                selected = []
        else:
            # 4. NEW RETRIEVAL
            candidates = self.retrieval.search(
                analyzed_query=analyzed,
                memory=session.get("memory"),
                top_k=10
            )

            # Save cache
            session["candidates"] = candidates
            session["current_index"] = 1
            session["last_query_signature"] = self._build_query_signature(analyzed)

            selected = candidates[:2]  # devolver 2 inicialmente

        # 5. UPDATE MEMORY
        self._update_memory(session, selected)

        # 6. BUILD PROMPT
        prompt = self.prompt_builder.build(
            user_input=user_input,
            context=selected,
            history=session.get("messages", [])
        )

        # 7. LLM CALL
        response = self.llm.generate(prompt)

        # 8. SAVE MESSAGE
        self.session_manager.add_message(session_id, "user", user_input)
        self.session_manager.add_message(session_id, "assistant", response)

        latency = time.time() - start_time

        return response, latency

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
    # QUERY SIGNATURE
    # =====================================================
    def _build_query_signature(self, analyzed):
        base = (
            analyzed.get("title", "") or "" +
            " ".join(analyzed.get("genres", [])) +
            " ".join(analyzed.get("keywords", []))
        )

        return hashlib.md5(base.encode()).hexdigest()

    # =====================================================
    # MEMORY UPDATE
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