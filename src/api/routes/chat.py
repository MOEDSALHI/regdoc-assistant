# src/api/routes/chat.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.services.llm_client import chat_stream

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7
    max_tokens: int = 1024


@router.post("/chat/stream")
async def stream_chat(request: ChatRequest) -> StreamingResponse:
    """
    Stream a chat response from Mistral AI using Server-Sent Events (SSE).

    Each chunk is prefixed with 'data: ' per the SSE spec.
    The stream ends with 'data: [DONE]'.
    """
    messages = [{"role": "user", "content": request.message}]

    async def event_generator():
        async for chunk in chat_stream(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disables nginx buffering in production
        },
    )
