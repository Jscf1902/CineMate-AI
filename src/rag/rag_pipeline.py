class RAGPipeline:
    def __init__(self):
        pass

    # =====================================================
    # MAIN
    # =====================================================
    def build_context(self, candidates):
        """
        Construye contexto textual para el LLM a partir de candidatos
        """

        if not candidates:
            return self._build_empty_context()

        context_blocks = []

        for item in candidates:
            block = self._format_item(item)
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    # =====================================================
    # FORMAT ITEM
    # =====================================================
    def _format_item(self, item):
        title = item.get("title", "Unknown")

        genres = ", ".join(item.get("genres", []))
        keywords = ", ".join(item.get("keywords", []))

        return (
            f"Película: {title}\n"
            f"Géneros: {genres}\n"
            f"Keywords: {keywords}"
        )

    # =====================================================
    # EMPTY CONTEXT
    # =====================================================
    def _build_empty_context(self):
        return "No se encontraron recomendaciones relevantes."