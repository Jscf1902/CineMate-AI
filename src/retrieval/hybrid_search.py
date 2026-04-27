import numpy as np # type: ignore
from collections import Counter

class HybridSearch:
    def __init__(self, faiss_index, metadata, embeddings_model):
        self.index = faiss_index
        self.metadata = metadata
        self.model = embeddings_model

        # 1. Pre-procesar el corpus y tokens una sola vez (Ahorra CPU en cada búsqueda)
        self.corpus_tokenized = []
        self.corpus_counts = []
        self.corpus_text_lower = [] # Para búsqueda rápida de strings
        
        for m in metadata:
            text = (
                f"{m.get('title', '')} {m.get('overview', '')} "
                f"{self._safe_join(m.get('keywords'))} {self._safe_join(m.get('genres'))}"
            ).lower()
            tokens = text.split()
            self.corpus_text_lower.append(text)
            self.corpus_tokenized.append(tokens)
            self.corpus_counts.append(Counter(tokens))

        # 2. Pre-calcular IDF y mapeo de títulos
        self.N = len(metadata)
        self.doc_freq = self._compute_doc_freq()
        self.idf_map = {t: np.log((self.N + 1) / (df + 1)) for t, df in self.doc_freq.items()}
        self.title_map = {str(m.get("title", "")).lower(): i for i, m in enumerate(metadata)}

    def search(self, analyzed_query, memory=None, top_k=10):
        intent = analyzed_query["intent_type"]
        
        # Optimización de lógica de base_query
        if intent == "TITLE":
            base_query = self._build_from_title(analyzed_query["title"])
        elif intent == "GENRE":
            base_query = " ".join(analyzed_query["genres"])
        else:
            base_query = " ".join(analyzed_query["keywords"])

        candidates = self._semantic_search(base_query, top_k=30)
        
        # Cachear sets para comparaciones rápidas en bucles
        query_keywords = analyzed_query.get("keywords", [])
        query_genres = analyzed_query.get("genres", [])
        seen_movies = set(memory.get("last_movies", [])) if memory else set()
        noise_words = ["novel", "book", "series"]

        scored = []
        for idx, sim_score in candidates:
            item = self.metadata[idx]
            
            # Usar datos pre-calculados
            lexical = self._lexical_score_optimized(query_keywords, idx)
            intent_score = self._intent_score_optimized(query_genres, query_keywords, idx)
            penalty = self._penalty_optimized(item, seen_movies, noise_words, idx)

            final_score = (0.5 * sim_score) + (0.3 * lexical) + (0.2 * intent_score) - penalty
            scored.append((item, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [self._format_output(item, score) for item, score in scored[:top_k]]

    # --- OPTIMIZACIONES DE MÉTODOS INTERNOS ---

    def _lexical_score_optimized(self, tokens, doc_idx):
        tf = self.corpus_counts[doc_idx]
        tokens_in_doc = self.corpus_tokenized[doc_idx]
        if not tokens_in_doc: return 0.0
        
        # Usar el idf_map pre-calculado
        score = sum(tf[t] * self.idf_map.get(t, 0.0) for t in tokens if t in tf)
        return score / (len(tokens_in_doc) + 1)

    def _intent_score_optimized(self, genres, keywords, idx):
        score = 0.0
        text = self.corpus_text_lower[idx]
        # Búsqueda de sub-strings más eficiente
        score += sum(1.0 for g in genres if g in text)
        score += sum(0.3 for kw in keywords if kw in text)
        return score

    def _penalty_optimized(self, item, seen_movies, noise_words, idx):
        penalty = 0.0
        if item.get("title") in seen_movies:
            penalty += 2.0
        
        text = self.corpus_text_lower[idx] # Usamos el texto ya procesado
        for n in noise_words:
            if n in text:
                penalty += 1.0
        return penalty

    def _build_from_title(self, title):
        # Búsqueda O(1) en lugar de O(n)
        idx = self.title_map.get(str(title).lower())
        if idx is not None:
            m = self.metadata[idx]
            return f"{self._safe_join(m.get('keywords'))} {self._safe_join(m.get('genres'))}"
        return title or ""

    # --- MÉTODOS DE APOYO ---

    def _compute_doc_freq(self):
        df = Counter()
        for tokens in self.corpus_tokenized:
            df.update(set(tokens))
        return df

    def _semantic_search(self, query, top_k=30):
        # El modelo suele ser lo más lento; normalize_embeddings fuera si el índice ya es Inner Product
        emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(emb, top_k)
        return [(i, float(s)) for i, s in zip(indices[0], scores[0]) if i != -1]

    def _safe_join(self, value):
        if isinstance(value, list):
            return " ".join(map(str, value))
        return str(value) if value else ""

    def _format_output(self, item, score):
        return {
            "title": item.get("title"),
            "score": round(score, 4),
            "genres": item.get("genres", []),
            "keywords": item.get("keywords", [])
        }