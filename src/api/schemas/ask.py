# src/api/schemas/ask.py
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="User question in natural language.",
        examples=["How long should access logs be stored under GDPR?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of document chunks to retrieve (1-10).",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.1=factual, 0.7=creative).",
    )


class Citation(BaseModel):
    document: str
    page: int | None
    quote: str


class AskResponse(BaseModel):
    answer: str | None
    confidence: str  # HIGH | MEDIUM | LOW
    citations: list[Citation]
    cannot_answer: bool
    chunks_used: int
    question: str