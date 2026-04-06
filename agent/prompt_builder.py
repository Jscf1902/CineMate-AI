class PromptBuilder:
    def __init__(self):
        pass

    # =====================================================
    # MAIN
    # =====================================================
    def build(self, user_input, context, history=None):
        """
        Construye prompt controlado para evitar alucinaciones
        """

        system_rules = self._build_system_rules()
        context_block = self._build_context_block(context)
        user_block = self._build_user_block(user_input)

        prompt = (
            f"{system_rules}\n\n"
            f"{context_block}\n\n"
            f"{user_block}"
        )

        return prompt

    # =====================================================
    # SYSTEM RULES
    # =====================================================
    def _build_system_rules(self):
        return (
            "Eres CineMate, un asistente de recomendación de películas.\n"
            "Debes seguir estrictamente estas reglas:\n"
            "- SOLO puedes recomendar películas que estén en el contexto.\n"
            "- NO inventes películas.\n"
            "- NO uses conocimiento externo.\n"
            "- Si no hay recomendaciones, usa el campo fallback.\n"
            "- Responde únicamente en formato JSON.\n"
            "- Máximo 3 recomendaciones.\n"
        )

    # =====================================================
    # CONTEXT BLOCK
    # =====================================================
    def _build_context_block(self, context):
        if not context:
            return "CONTEXTO:\nNo hay recomendaciones disponibles."

        lines = ["CONTEXTO:"]
        for item in context:
            title = item.get("title")
            genres = ", ".join(item.get("genres", []))
            keywords = ", ".join(item.get("keywords", []))

            lines.append(
                f"- {title} | géneros: {genres} | keywords: {keywords}"
            )

        return "\n".join(lines)

    # =====================================================
    # USER BLOCK
    # =====================================================
    def _build_user_block(self, user_input):
        return f"USUARIO:\n{user_input}"
