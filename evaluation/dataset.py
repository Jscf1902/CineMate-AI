import os
import json
from typing import List, Dict


class EvaluationDataset:
    def __init__(self, interactions_path: str = "interactions"):
        self.path = interactions_path

    # =====================================================
    # LOAD ALL SESSIONS
    # =====================================================
    def load_sessions(self) -> List[Dict]:
        sessions = []

        if not os.path.exists(self.path):
            return sessions

        for file in os.listdir(self.path):
            if not file.endswith(".json"):
                continue

            full_path = os.path.join(self.path, file)

            with open(full_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    sessions.append(data)
                except Exception:
                    continue

        return sessions

    # =====================================================
    # FLATTEN INTERACTIONS
    # =====================================================
    def load_interactions(self) -> List[Dict]:
        sessions = self.load_sessions()
        interactions = []

        for session in sessions:
            session_id = session.get("session_id")
            mode = session.get("mode")

            for inter in session.get("interactions", []):
                interactions.append({
                    "session_id": session_id,
                    "mode": mode,
                    "query": inter.get("query"),
                    "routed_query": inter.get("routed_query"),
                    "latency": inter.get("latency"),
                    "used_cache": inter.get("used_cache"),
                    "scores": inter.get("scores", []),
                    "titles": inter.get("titles", []),
                    "results_count": inter.get("results_count", 0)
                })

        return interactions

    # =====================================================
    # LOAD FEEDBACK (CSAT / NPS / RESOLUTION)
    # =====================================================
    def load_feedback(self) -> List[Dict]:
        sessions = self.load_sessions()
        feedback_data = []

        for session in sessions:
            session_id = session.get("session_id")
            mode = session.get("mode")

            csat = session.get("csat", {})
            nps = session.get("nps", {})
            resolution = session.get("resolution", {})

            # ignorar sesiones sin feedback
            if not csat and not nps and not resolution:
                continue

            feedback_data.append({
                "session_id": session_id,
                "mode": mode,
                "csat_score": csat.get("score"),
                "nps_score": nps.get("score"),
                "nps_category": nps.get("category"),
                "resolution_score": resolution.get("numeric"),
                "resolution_label": resolution.get("label")
            })

        return feedback_data

    # =====================================================
    # SPLIT BY MODE
    # =====================================================
    def split_by_mode(self, data: List[Dict]):
        rag = []
        direct = []

        for item in data:
            if item.get("mode") == "RAG":
                rag.append(item)
            else:
                direct.append(item)

        return rag, direct

    # =====================================================
    # BASIC STATS
    # =====================================================
    def summary(self, interactions: List[Dict]) -> Dict:
        total = len(interactions)

        rag_count = sum(1 for i in interactions if i["mode"] == "RAG")
        direct_count = total - rag_count

        cache_count = sum(1 for i in interactions if i.get("used_cache"))

        return {
            "total_interactions": total,
            "rag": rag_count,
            "direct": direct_count,
            "cache_usage": cache_count
        }

    # =====================================================
    # FEEDBACK SUMMARY
    # =====================================================
    def feedback_summary(self, feedback: List[Dict]) -> Dict:
        total = len(feedback)

        if total == 0:
            return {
                "total_feedback": 0
            }

        avg_csat = sum(f["csat_score"] for f in feedback if f["csat_score"] is not None) / total

        avg_nps = sum(f["nps_score"] for f in feedback if f["nps_score"] is not None) / total

        avg_resolution = sum(f["resolution_score"] for f in feedback if f["resolution_score"] is not None) / total

        promoters = sum(1 for f in feedback if f["nps_category"] == "promoter")
        detractors = sum(1 for f in feedback if f["nps_category"] == "detractor")

        nps_value = (promoters / total) - (detractors / total)

        return {
            "total_feedback": total,
            "csat_avg": round(avg_csat, 4),
            "nps_avg": round(avg_nps, 4),
            "nps_score": round(nps_value, 4),
            "resolution_avg": round(avg_resolution, 4)
        }