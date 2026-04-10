from fastapi import APIRouter, HTTPException
from api.schemas import ChatRequest, ChatResponse
from api.dependencies import get_orchestrator

router = APIRouter()

orchestrator = get_orchestrator()


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        # -------------------------
        # SESSION ID
        # -------------------------
        session_id = request.session_id
        session = orchestrator.session_manager.get_session(session_id)
        session["user_id"] = request.user_id
        orchestrator.session_manager.save_session(session)

        # -------------------------
        # INIT MODE
        # -------------------------
        orchestrator.init_session_mode(session_id)

        # -------------------------
        # ORCHESTRATOR
        # -------------------------
        response, latency, mode = orchestrator.handle_message(
            session_id=session_id,
            user_input=request.message
        )

        return ChatResponse(response=response)

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error procesando la solicitud"
        )