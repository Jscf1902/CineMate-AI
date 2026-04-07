import datetime


class FeedbackCollector:
    def __init__(self, session_manager):
        self.session_manager = session_manager

    # =====================================================
    # MAIN
    # =====================================================
    def run(self, session_id):
        session = self.session_manager.get_session(session_id)

        print("\n--- Encuesta de satisfacción ---\n")

        csat = self._ask_csat()
        nps = self._ask_nps()
        resolution = self._ask_resolution()

        feedback_data = {
            "csat": {
                "score": csat,
                "timestamp": datetime.datetime.now().isoformat()
            },
            "nps": {
                "score": nps,
                "category": self._classify_nps(nps)
            },
            "resolution": {
                "label": resolution["label"],
                "numeric": resolution["numeric"]
            }
        }

        self.session_manager.save_feedback(session, feedback_data)

        print("\nGracias por tu feedback.\n")

    # =====================================================
    # CSAT
    # =====================================================
    def _ask_csat(self):
        while True:
            try:
                value = int(input(
                    "1. ¿Qué tan satisfecho estás con la recomendación? (1–5): "
                ))
                if 1 <= value <= 5:
                    return value
            except:
                pass

            print("Entrada inválida. Ingresa un número entre 1 y 5.")

    # =====================================================
    # NPS
    # =====================================================
    def _ask_nps(self):
        while True:
            try:
                value = int(input(
                    "2. ¿Qué tan probable es que recomiendes el chatbot? (0–10): "
                ))
                if 0 <= value <= 10:
                    return value
            except:
                pass

            print("Entrada inválida. Ingresa un número entre 0 y 10.")

    def _classify_nps(self, score):
        if score <= 6:
            return "detractor"
        elif score <= 8:
            return "passive"
        else:
            return "promoter"

    # =====================================================
    # RESOLUTION
    # =====================================================
    def _ask_resolution(self):
        print("\n3. ¿La recomendación se ajustó a tus gustos?")
        print("1. Sí, totalmente")
        print("2. Parcialmente")
        print("3. No")

        while True:
            value = input("Selecciona una opción (1–3): ")

            if value == "1":
                return {"label": "yes", "numeric": 3}
            elif value == "2":
                return {"label": "partial", "numeric": 2}
            elif value == "3":
                return {"label": "no", "numeric": 1}

            print("Entrada inválida. Ingresa 1, 2 o 3.")