import numpy as np
from typing import List, Dict


class Metrics:

    # =====================================================
    # LATENCY
    # =====================================================
    @staticmethod
    def latency_avg(data: List[Dict]) -> float:
        values = [d["latency"] for d in data if d.get("latency") is not None]
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def latency_p95(data: List[Dict]) -> float:
        values = [d["latency"] for d in data if d.get("latency") is not None]
        return float(np.percentile(values, 95)) if values else 0.0

    # =====================================================
    # HIT RATE
    # =====================================================
    @staticmethod
    def hit_rate(data: List[Dict], threshold: float = 0.75) -> float:
        hits = 0

        for d in data:
            scores = d.get("scores", [])
            if any(s >= threshold for s in scores):
                hits += 1

        return hits / len(data) if data else 0.0

    # =====================================================
    # MRR
    # =====================================================
    @staticmethod
    def mrr(data: List[Dict], threshold: float = 0.75) -> float:
        total = 0.0

        for d in data:
            scores = d.get("scores", [])

            rank = 0
            for i, s in enumerate(scores, start=1):
                if s >= threshold:
                    rank = i
                    break

            if rank > 0:
                total += 1.0 / rank

        return total / len(data) if data else 0.0

    # =====================================================
    # SCORE
    # =====================================================
    @staticmethod
    def score_avg(data: List[Dict]) -> float:
        scores = []
        for d in data:
            scores.extend(d.get("scores", []))
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def score_top1(data: List[Dict]) -> float:
        scores = []
        for d in data:
            s = d.get("scores", [])
            if s:
                scores.append(s[0])
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def score_gap(data: List[Dict]) -> float:
        gaps = []
        for d in data:
            s = d.get("scores", [])
            if len(s) >= 2:
                gaps.append(s[0] - s[1])
        return float(np.mean(gaps)) if gaps else 0.0

    # =====================================================
    # DIVERSITY
    # =====================================================
    @staticmethod
    def diversity(data: List[Dict]) -> float:
        diffs = []
        for d in data:
            s = d.get("scores", [])
            if len(s) >= 2:
                diffs.append(abs(s[0] - s[1]))
        return float(np.mean(diffs)) if diffs else 0.0

    # =====================================================
    # CACHE
    # =====================================================
    @staticmethod
    def cache_usage(data: List[Dict]) -> float:
        count = sum(1 for d in data if d.get("used_cache"))
        return count / len(data) if data else 0.0

    # =====================================================
    # CSAT
    # =====================================================
    @staticmethod
    def csat_avg(data: List[Dict]) -> float:
        values = [d["csat_score"] for d in data if d.get("csat_score") is not None]
        return float(np.mean(values)) if values else 0.0

    # =====================================================
    # NPS
    # =====================================================
    @staticmethod
    def nps_score(data: List[Dict]) -> float:
        if not data:
            return 0.0

        promoters = sum(1 for d in data if d.get("nps_category") == "promoter")
        detractors = sum(1 for d in data if d.get("nps_category") == "detractor")

        return (promoters / len(data)) - (detractors / len(data))

    # =====================================================
    # RESOLUTION
    # =====================================================
    @staticmethod
    def resolution_avg(data: List[Dict]) -> float:
        values = [d["resolution_score"] for d in data if d.get("resolution_score") is not None]
        return float(np.mean(values)) if values else 0.0

    # =====================================================
    # FULL EVALUATION
    # =====================================================
    @staticmethod
    def evaluate_internal(data: List[Dict]) -> Dict:
        return {
            "latency_avg": Metrics.latency_avg(data),
            "latency_p95": Metrics.latency_p95(data),
            "hit_rate": Metrics.hit_rate(data),
            "mrr": Metrics.mrr(data),
            "score_avg": Metrics.score_avg(data),
            "score_top1": Metrics.score_top1(data),
            "score_gap": Metrics.score_gap(data),
            "diversity": Metrics.diversity(data),
            "cache_usage": Metrics.cache_usage(data)
        }

    @staticmethod
    def evaluate_feedback(data: List[Dict]) -> Dict:
        return {
            "csat_avg": Metrics.csat_avg(data),
            "nps": Metrics.nps_score(data),
            "resolution_avg": Metrics.resolution_avg(data)
        }