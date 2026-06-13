# src/api/routes/ingest.py
from fastapi import APIRouter

from src.api.schemas.ingest import IngestRequest, IngestResponse
from src.rag.ingestion import ingest_document

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """
    Ingest a text document into pgvector.

    Chunks the text, generates embeddings, and stores everything.
    Idempotent: re-ingesting the same content is a no-op unless force_reingest=True.
    """
    summary = await ingest_document(
        source=request.text,
        doc_type=request.doc_type,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
        force_reingest=request.force_reingest,
        filename=request.filename,
    )
    return IngestResponse(**summary)
