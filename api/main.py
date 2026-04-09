from fastapi import FastAPI
from api.routes import router
from api.feedback import router as feedback_router

app = FastAPI(
    title="CineMate AI API",
    version="1.0.0"
)

# rutas existentes
app.include_router(router)

# nuevo endpoint de feedback
app.include_router(feedback_router)


@app.get("/")
def root():
    return {"message": "CineMate AI API running"}