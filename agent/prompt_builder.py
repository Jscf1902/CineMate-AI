class PromptBuilder:

    # =====================================================
    # MAIN
    # =====================================================
    def build(self, user_input, context, history=None):

        system_rules = self._build_system_rules()
        history_block = self._build_history_block(history)
        context_block = self._build_context_block(context)
        user_block = self._build_user_block(user_input)

        return (
            f"{system_rules}\n\n"
            f"{history_block}\n\n"
            f"{context_block}\n\n"
            f"{user_block}\n\n"
            f"RESPUESTA:"
        )

    # =====================================================
    # SAFE JOIN
    # =====================================================
    def _safe_join(self, value):
        if isinstance(value, list):
            return ", ".join([str(v) for v in value if v])
        if isinstance(value, str):
            return value
        return ""

    # =====================================================
    # SYSTEM RULES (MEJORADO)
    # =====================================================
    def _build_system_rules(self):
        return (
            "You are CineMate, a movie recommendation assistant.\n\n"

            "## Core Rules:\n"
            "- ONLY use the provided context.\n"
            "- NEVER invent or hallucinate movie titles.\n"
            "- If the context is weak or irrelevant, say you don't have enough information.\n"
            "- Make minimum 5 recommendations.\n"
            "- Keep answers concise and useful.\n"
            "- Detect the language of the user and respond in that same language.\n\n"

            "## Conversation Behavior:\n"
            "- If the user says 'sí' → continue previous recommendation.\n"
            "- If the user says 'otra' → give different recommendations.\n"
            "- If the user says 'ya la vi' → avoid repeating and change suggestions.\n"
            "- Use conversation history to maintain context.\n\n"

            "## Output:\n"
            "- Respond in natural conversational text.\n"
            "- Do NOT output JSON.\n"
        )

    # =====================================================
    # HISTORY
    # =====================================================
    def _build_history_block(self, history):
        if not history:
            return "HISTORIAL:\nSin historial."

        return f"HISTORIAL:\n{history}"

    # =====================================================
    # CONTEXT
    # =====================================================
    def _build_context_block(self, context):

        if not context or len(context) == 0:
            return "CONTEXTO:\nNo hay resultados relevantes."

        lines = ["CONTEXTO:"]

        for item in context:

            title = str(item.get("title", ""))
            overview = str(item.get("overview", ""))

            genres = self._safe_join(item.get("genres"))
            keywords = self._safe_join(item.get("keywords"))

            lines.append(
                f"- {title}\n"
                f"  géneros: {genres}\n"
                f"  keywords: {keywords}\n"
                f"  resumen: {overview}"
            )

        return "\n".join(lines)

    # =====================================================
    # USER
    # =====================================================
    def _build_user_block(self, user_input):
        return f"USUARIO:\n{user_input}"