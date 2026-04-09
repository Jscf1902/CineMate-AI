from fastapi import APIRouter
from datetime import datetime

from agent.session_manager import SessionManager

router = APIRouter()

session_manager = SessionManager()


@router.post("/feedback")
def receive_feedback(data: dict):

    session_id = data.get("session_id")

    if not session_id:
        return {"status": "error", "message": "session_id is required"}

    session = session_manager.get_session(session_id)

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

    session_manager.save_feedback(session, feedback_data)

    return {"status": "ok"}