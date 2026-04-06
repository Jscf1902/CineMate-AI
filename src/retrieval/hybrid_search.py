import numpy as np
from collections import Counter


class HybridSearch:
    def __init__(self, faiss_index, metadata, embeddings_model):
        self.index = faiss_index
        self.metadata = metadata
        self.model = embeddings_model

        # Corpus robusto (maneja listas, strings, None)
        self.corpus = [
            (
                str(m.get("title", "")) + " " +
                str(m.get("overview", "")) + " " +
                self._safe_join(m.get("keywords")) + " " +
                self._safe_join(m.get("genres"))
            ).lower()
            for m in metadata
        ]

        self.doc_freq = self._compute_doc_freq()
        self.N = len(self.corpus)

    # =====================================================
    # MAIN
    # =====================================================
    def search(self, analyzed_query, memory=None, top_k=10):
        intent = analyzed_query["intent_type"]

        if intent == "TITLE":
            base_query = self._build_from_title(analyzed_query["title"])
        elif intent == "GENRE":
            base_query = " ".join(analyzed_query["genres"])
        else:
            base_query = " ".join(analyzed_query["keywords"])

        candidates = self._semantic_search(base_query, top_k=30)

        scored = []
        for idx, sim_score in candidates:
            item = self.metadata[idx]

            lexical = self._lexical_score(analyzed_query["keywords"], idx)
            intent_score = self._intent_score(analyzed_query, item)
            penalty = self._penalty(item, memory)

            final_score = (
                0.5 * sim_score +
                0.3 * lexical +
                0.2 * intent_score -
                penalty
            )

            scored.append((item, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            self._format_output(item, score)
            for item, score in scored[:top_k]
        ]

    # =====================================================
    # SAFE JOIN (ROBUSTO)
    # =====================================================
    def _safe_join(self, value):
        if isinstance(value, list):
            return " ".join([str(v) for v in value if v])
        if isinstance(value, str):
            return value
        return ""

    # =====================================================
    # TITLE STRATEGY
    # =====================================================
    def _build_from_title(self, title):
        for m in self.metadata:
            if str(m.get("title", "")).lower() == str(title).lower():
                return self._safe_join(m.get("keywords")) + " " + self._safe_join(m.get("genres"))
        return title or ""

    # =====================================================
    # SEMANTIC SEARCH
    # =====================================================
    def _semantic_search(self, query, top_k=30):
        emb = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        emb = np.array([emb], dtype="float32")

        scores, indices = self.index.search(emb, top_k)

        results = []
        for i, s in zip(indices[0], scores[0]):
            if i != -1:
                results.append((i, float(s)))

        return results

    # =====================================================
    # LEXICAL SCORING
    # =====================================================
    def _compute_doc_freq(self):
        df = Counter()
        for doc in self.corpus:
            for t in set(doc.split()):
                df[t] += 1
        return df

    def _lexical_score(self, tokens, doc_idx):
        doc = self.corpus[doc_idx].split()
        tf = Counter(doc)

        score = 0.0
        for t in tokens:
            if t not in tf:
                continue

            df = self.doc_freq.get(t, 1)
            idf = np.log((self.N + 1) / (df + 1))

            score += tf[t] * idf

        return score / (len(doc) + 1)

    # =====================================================
    # INTENT SCORE
    # =====================================================
    def _intent_score(self, analyzed, item):
        score = 0.0

        text = (
            str(item.get("title", "")).lower() + " " +
            str(item.get("overview", "")).lower() + " " +
            self._safe_join(item.get("keywords")).lower()
        )

        for g in analyzed["genres"]:
            if g in text:
                score += 1.0

        for kw in analyzed["keywords"]:
            if kw in text:
                score += 0.3

        return score

    # =====================================================
    # PENALTY
    # =====================================================
    def _penalty(self, item, memory):
        penalty = 0.0

        if not memory:
            return penalty

        seen = memory.get("last_movies", [])
        if item.get("title") in seen:
            penalty += 2.0

        text = str(item.get("overview", "")).lower()
        noise = ["novel", "book", "series"]

        for n in noise:
            if n in text:
                penalty += 1.0

        return penalty

    # =====================================================
    # OUTPUT
    # =====================================================
    def _format_output(self, item, score):
        return {
            "title": item.get("title"),
            "score": round(score, 4),
            "genres": item.get("genres", []),
            "keywords": item.get("keywords", [])
        }