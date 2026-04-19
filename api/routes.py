from fastapi import APIRouter, HTTPException
from api.schemas import ChatRequest, ChatResponse
from api.dependencies import get_orchestrator

from datetime import datetime
import time


router = APIRouter()

# singleton
orchestrator = get_orchestrator()

# métricas simples
TOTAL_REQUESTS = 0


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):

    global TOTAL_REQUESTS

    start_time = time.time()
    TOTAL_REQUESTS += 1

    request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # -------------------------
        # VALIDACIONES
        # -------------------------
        if not request.session_id:
            raise HTTPException(
                status_code=400,
                detail="session_id requerido"
            )

        if not request.user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id requerido"
            )

        if not request.message or not request.message.strip():
            raise HTTPException(
                status_code=400,
                detail="message requerido"
            )

        # -------------------------
        # LOG INICIO
        # -------------------------
        print("\n===================================")
        print("Nueva request /chat")
        print("Hora:", request_time)
        print("Request #:", TOTAL_REQUESTS)
        print("Session:", request.session_id)
        print("User:", request.user_id)
        print("Message:", request.message)
        print("===================================\n")

        # -------------------------
        # SESSION
        # -------------------------
        session_id = request.session_id

        session = orchestrator.session_manager.get_session(
            session_id
        )

        session["user_id"] = request.user_id

        orchestrator.session_manager.save_session(
            session
        )

        # -------------------------
        # INIT MODE
        # -------------------------
        orchestrator.init_session_mode(session_id)

        # -------------------------
        # HANDLE MESSAGE
        # -------------------------
        response, latency, mode = (
            orchestrator.handle_message(
                session_id=session_id,
                user_input=request.message
            )
        )

        # -------------------------
        # LOG FIN
        # -------------------------
        total_ms = int(
            (time.time() - start_time) * 1000
        )

        print("\n-----------------------------------")
        print("Request completada")
        print("Session:", session_id)
        print("Modo:", mode)
        print("LLM latency:", latency, "ms")
        print("Total API:", total_ms, "ms")
        print("-----------------------------------\n")

        return ChatResponse(
            response=response
        )

    except HTTPException as e:
        raise e

    except Exception as e:

        total_ms = int(
            (time.time() - start_time) * 1000
        )

        print("\n******** ERROR /chat ********")
        print("Session:", request.session_id)
        print("User:", request.user_id)
        print("Tiempo:", total_ms, "ms")
        print("Detalle:", str(e))
        print("****************************\n")

        raise HTTPException(
            status_code=500,
            detail="Error procesando la solicitud"
        )