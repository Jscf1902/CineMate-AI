class PromptBuilder:
    def __init__(self):
        pass

    # =====================================================
    # MAIN
    # =====================================================
    def build(self, user_input, context, history=None):
        system_rules = self._build_system_rules()
        context_block = self._build_context_block(context)
        user_block = self._build_user_block(user_input)

        return (
            f"{system_rules}\n\n"
            f"{context_block}\n\n"
            f"{user_block}"
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
    # SYSTEM RULES
    # =====================================================
    def _build_system_rules(self):
        return (
            "Eres CineMate, un asistente de recomendación.\n"
            "- SOLO usa el contexto.\n"
            "- NO inventes películas.\n"
            "- Máximo 5 recomendaciones.\n"
            "- Responde en texto natural.\n"
        )

    # =====================================================
    # CONTEXT
    # =====================================================
    def _build_context_block(self, context):
        if not context:
            return "CONTEXTO:\nNo hay recomendaciones disponibles."

        lines = ["CONTEXTO:"]

        for item in context:
            title = str(item.get("title", ""))

            genres = self._safe_join(item.get("genres"))
            keywords = self._safe_join(item.get("keywords"))

            lines.append(
                f"- {title} | géneros: {genres} | keywords: {keywords}"
            )

        return "\n".join(lines)

    # =====================================================
    # USER
    # =====================================================
    def _build_user_block(self, user_input):
        return f"USUARIO:\n{user_input}"