from __future__ import annotations

from llm.types import ChatMessage, ChatResponse


class MockProvider:
    """Deterministic, no-network provider. Always registered, needs no API key.

    This is the default LLM_PROVIDER so `docker compose up`, CI, and the demo
    route all work with zero API keys out of the box.
    """

    name = "mock"

    async def chat_completion(
        self, messages: list[ChatMessage], model: str | None = None, **kwargs: object
    ) -> ChatResponse:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return ChatResponse(
            text=f"mock reply: {last_user}",
            provider=self.name,
            model=model or "mock-echo",
            tokens_in=len(last_user.split()),
            tokens_out=len(last_user.split()) + 2,
            cost_usd=0.0,
        )
