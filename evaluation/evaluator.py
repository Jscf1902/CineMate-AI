from evaluation.dataset import EvaluationDataset
from evaluation.metrics import Metrics


class Evaluator:
    def __init__(self, path="interactions"):
        self.dataset = EvaluationDataset(path)

    # =====================================================
    # MAIN
    # =====================================================
    def run(self):

        interactions = self.dataset.load_interactions()
        feedback = self.dataset.load_feedback()

        if not interactions:
            print("No hay datos.")
            return

        rag_i, direct_i = self.dataset.split_by_mode(interactions)
        rag_f, direct_f = self.dataset.split_by_mode(feedback)

        print("\n==============================")
        print("INTERNAL METRICS")
        print("==============================")

        print("\nGLOBAL")
        self._print(Metrics.evaluate_internal(interactions))

        print("\nRAG")
        rag_internal = Metrics.evaluate_internal(rag_i)
        self._print(rag_internal)

        print("\nDIRECT")
        direct_internal = Metrics.evaluate_internal(direct_i)
        self._print(direct_internal)

        print("\n==============================")
        print("FEEDBACK METRICS")
        print("==============================")

        print("\nGLOBAL")
        self._print(Metrics.evaluate_feedback(feedback))

        print("\nRAG")
        rag_feedback = Metrics.evaluate_feedback(rag_f)
        self._print(rag_feedback)

        print("\nDIRECT")
        direct_feedback = Metrics.evaluate_feedback(direct_f)
        self._print(direct_feedback)

        print("\n==============================")
        print("COMPARISON")
        print("==============================")

        self._compare(rag_internal, direct_internal)

    # =====================================================
    # PRINT
    # =====================================================
    def _print(self, metrics):
        for k, v in metrics.items():
            print(f"{k}: {round(v, 4)}")

    # =====================================================
    # COMPARE
    # =====================================================
    def _compare(self, rag, direct):
        for k in rag.keys():
            r = rag[k]
            d = direct[k]
            print(f"{k}: RAG={round(r,4)} | DIRECT={round(d,4)} | diff={round(r-d,4)}")