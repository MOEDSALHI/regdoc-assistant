# src/api/routes/health.py
from fastapi import APIRouter
from pydantic import BaseModel

from src.config import settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    env: str
    model: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe — confirms the API is running and config is loaded."""
    return HealthResponse(
        status="ok",
        env=settings.app_env,
        model=settings.mistral_model,
    )
