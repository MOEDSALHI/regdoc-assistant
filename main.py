# main.py
import uvicorn
from fastapi import FastAPI
from loguru import logger

from src.config import settings
from src.api.routes.health import router as health_router
from src.api.routes.chat import router as chat_router

app = FastAPI(
    title="RAG Project",
    description="Document assistant with citations — powered by Mistral AI + pgvector",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(chat_router)


@app.on_event("startup")
async def startup():
    logger.info("Starting RAG API | env={} | model={}", settings.app_env, settings.mistral_model)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)