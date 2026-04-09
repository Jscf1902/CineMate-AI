from fastapi import APIRouter
from datetime import datetime

from api.dependencies import get_orchestrator

router = APIRouter()


@router.post("/feedback")
def receive_feedback(data: dict):

    # -------------------------
    # ORCHESTRATOR (SINGLETON)
    # -------------------------
    orchestrator = get_orchestrator()
    session_manager = orchestrator.session_manager

    # -------------------------
    # SESSION
    # -------------------------
    session_id = data.get("session_id")

    if not session_id:
        return {"status": "error", "message": "session_id is required"}

    session = session_manager.get_session(session_id)

    # -------------------------
    # BUILD FEEDBACK
    # -------------------------
    feedback_data = {
        "csat": {
            "score": data["feedback"].get("csat"),
            "timestamp": datetime.now().isoformat()
        },
        "nps": {
            "score": data["feedback"].get("nps"),
            "category": data["feedback"].get("nps_category")
        },
        "resolution": data["feedback"].get("resolution"),
        "query": data.get("query"),
        "recommendation": data.get("recommendation")
    }

    # -------------------------
    # SAVE
    # -------------------------
    session_manager.save_feedback(session, feedback_data)

    print("Feedback guardado correctamente")

    return {"status": "ok"}