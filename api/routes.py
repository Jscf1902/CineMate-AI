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
        session_id = f"whatsapp_{request.user_id}"
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