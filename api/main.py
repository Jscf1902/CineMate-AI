from fastapi import FastAPI
from api.routes import router
from api.feedback import router as feedback_router

from datetime import datetime
import time


# -------------------------
# STARTUP TIME
# -------------------------
START_TIME = time.time()


# -------------------------
# APP
# -------------------------
app = FastAPI(
    title="CineMate AI API",
    version="1.0.0"
)


# -------------------------
# ROUTERS
# -------------------------
app.include_router(router)
app.include_router(feedback_router)


# -------------------------
# ROOT
# -------------------------
@app.get("/")
def root():
    return {
        "message": "CineMate AI API running",
        "status": "online",
        "version": "1.0.0"
    }


# -------------------------
# STATUS
# -------------------------
@app.get("/status")
def status():

    uptime_seconds = int(
        time.time() - START_TIME
    )

    uptime_minutes = round(
        uptime_seconds / 60, 2
    )

    return {
        "app": "CineMate AI API",
        "status": "online",
        "version": "1.0.0",
        "server_time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "uptime_seconds": uptime_seconds,
        "uptime_minutes": uptime_minutes
    }


# -------------------------
# STARTUP LOG
# -------------------------
@app.on_event("startup")
def startup_event():

    print("\n===================================")
    print("CineMate AI API iniciada")
    print("Hora:",
          datetime.now().strftime(
              "%Y-%m-%d %H:%M:%S"
          ))
    print("Version: 1.0.0")
    print("===================================\n")