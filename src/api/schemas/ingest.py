# src/api/schemas/ingest.py
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=100,
        description="Raw text to ingest (use for testing without PDF upload).",
    )
    doc_type: str = Field(
        default="regulatory",
        description="Document type: regulatory, guide, recommendation.",
    )
    chunk_size: int = Field(default=512, ge=64, le=1024)
    overlap: int = Field(default=64, ge=0, le=256)
    force_reingest: bool = Field(default=False)


class IngestResponse(BaseModel):
    doc_id: int
    filename: str
    chunks_count: int
    pages_processed: int
    avg_tokens_per_chunk: float
    latency_ms: int
    skipped: bool