from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from llm import ChatClient, ChatMessage, ProviderClientError, ProviderConfigError, ProviderError

router = APIRouter(prefix="/demo", tags=["demo"])

# Module-level singleton so retry/throttle state persists across requests within
# one process. max_retry_attempts is fixed at import time from Settings; the
# provider itself is still chosen fresh per-request from settings.llm_provider.
client = ChatClient(max_retry_attempts=get_settings().llm_max_retry_attempts)


class ChatRequest(BaseModel):
    prompt: str


class ChatReply(BaseModel):
    text: str
    provider: str
    model: str
    tokens_in: int | None
    tokens_out: int | None
    cost_usd: float | None
    latency_ms: float


@router.post("/chat", response_model=ChatReply)
async def chat(request: ChatRequest) -> ChatReply:
    settings = get_settings()
    try:
        response = await client.complete(
            settings.llm_provider,
            [ChatMessage(role="user", content=request.prompt)],
        )
    except ProviderConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ProviderClientError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatReply(**response.model_dump())
