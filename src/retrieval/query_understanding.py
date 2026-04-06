import re
from difflib import SequenceMatcher


class QueryUnderstanding:
    def __init__(self, metadata):
        """
        metadata: lista de películas (dicts)
        """
        self.metadata = metadata

        # Preprocesar títulos
        self.titles = [m["title"].lower() for m in metadata]

        # Catálogo básico de géneros
        self.GENRES = [
            "action", "comedy", "drama", "horror",
            "romance", "thriller", "anime", "fantasy",
            "adventure", "sci-fi", "shonen", "supernatural"
        ]

        # Stopwords simples
        self.STOPWORDS = {
            "peliculas", "pelicula", "quiero", "dame",
            "algo", "ver", "recomiendame", "recomendar",
            "parecido", "parecidas", "como", "de"
        }

    # =====================================================
    # MAIN
    # =====================================================
    def analyze(self, query, memory=None):
        query = query.lower().strip()

        title = self._detect_title(query)
        genres = self._detect_genres(query)
        keywords = self._extract_keywords(query)
        is_followup = self._is_followup(query)

        intent_type = self._classify_intent(title, genres, keywords)

        return {
            "original_query": query,
            "intent_type": intent_type,
            "title": title,
            "genres": genres,
            "keywords": keywords,
            "is_followup": is_followup
        }

    # =====================================================
    # TITLE DETECTION (MEJORADO)
    # =====================================================
    def _detect_title(self, query):
        """
        Busca coincidencias fuzzy con títulos del dataset
        """
        best_match = None
        best_score = 0.0

        for title in self.titles:
            score = SequenceMatcher(None, query, title).ratio()

            # threshold ajustado (evita ruido)
            if score > 0.65 and score > best_score:
                best_score = score
                best_match = title

        return best_match

    # =====================================================
    # GENRES
    # =====================================================
    def _detect_genres(self, query):
        return [g for g in self.GENRES if g in query]

    # =====================================================
    # KEYWORDS
    # =====================================================
    def _extract_keywords(self, query):
        tokens = re.findall(r"\w+", query)

        return [
            t for t in tokens
            if t not in self.STOPWORDS and len(t) > 2
        ]

    # =====================================================
    # FOLLOW-UP DETECTION
    # =====================================================
    def _is_followup(self, query):
        triggers = [
            "otra", "otro", "más", "mas",
            "diferente", "ya la vi", "otra opcion"
        ]
        return any(t in query for t in triggers)

    # =====================================================
    # INTENT CLASSIFICATION
    # =====================================================
    def _classify_intent(self, title, genres, keywords):
        """
        Prioridad:
        1. TITLE
        2. GENRE
        3. KEYWORD
        """
        if title:
            return "TITLE"
        if genres:
            return "GENRE"
        return "KEYWORD"