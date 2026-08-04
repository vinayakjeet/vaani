from __future__ import annotations

from app.routers import demo as demo_module
from llm.types import ProviderClientError, ProviderConfigError, ProviderError


def test_chat_returns_mock_reply(client):
    resp = client.post("/demo/chat", json={"prompt": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "mock"
    assert "hello" in body["text"]
    assert body["tokens_in"] is not None
    assert body["cost_usd"] == 0.0


def test_chat_maps_provider_error_to_502(client, monkeypatch):
    async def _boom(*args: object, **kwargs: object):
        raise ProviderError("upstream is down")

    monkeypatch.setattr(demo_module.client, "complete", _boom)
    resp = client.post("/demo/chat", json={"prompt": "hello"})
    assert resp.status_code == 502


def test_chat_maps_client_error_to_400(client, monkeypatch):
    async def _bad(*args: object, **kwargs: object):
        raise ProviderClientError("bad request upstream")

    monkeypatch.setattr(demo_module.client, "complete", _bad)
    resp = client.post("/demo/chat", json={"prompt": "hello"})
    assert resp.status_code == 400


def test_chat_maps_config_error_to_500(client, monkeypatch):
    async def _misconfigured(*args: object, **kwargs: object):
        raise ProviderConfigError("missing API key")

    monkeypatch.setattr(demo_module.client, "complete", _misconfigured)
    resp = client.post("/demo/chat", json={"prompt": "hello"})
    assert resp.status_code == 500
