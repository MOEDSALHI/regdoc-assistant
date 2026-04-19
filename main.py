# main.py
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger

from src.api.routes.ask import router as ask_router
from src.api.routes.chat import router as chat_router
from src.api.routes.health import router as health_router
from src.api.routes.ingest import router as ingest_router
from src.config import settings
from src.db.database import apply_schema, close_pool, init_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — runs setup before serving requests,
    and teardown when the server shuts down.

    Everything BEFORE yield = startup (runs once before first request).
    Everything AFTER yield  = shutdown (runs on SIGTERM or Ctrl+C).

    This replaces the deprecated @app.on_event("startup") pattern.
    """
    # STARTUP
    logger.info(
        "Starting RegDoc API | env={} | model={}",
        settings.app_env,
        settings.mistral_model,
    )
    await init_pool()       # initialize asyncpg connection pool
    await apply_schema()    # create tables/indexes if not exist (idempotent)
    logger.info("Database ready")

    yield  # <-- application serves requests here

    # SHUTDOWN
    await close_pool()
    logger.info("RegDoc API shutdown complete")


app = FastAPI(
    title="RegDoc Assistant",
    description="Regulatory document assistant — GDPR, CNIL, ANSSI",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(ask_router)
app.include_router(ingest_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)