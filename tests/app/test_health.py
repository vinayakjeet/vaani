from __future__ import annotations


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_version(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "0.1.0"
    assert "git_sha" in body
    assert "environment" in body


def test_response_carries_request_id_header(client):
    resp = client.get("/healthz")
    assert "X-Request-ID" in resp.headers


def test_incoming_request_id_is_echoed_back(client):
    resp = client.get("/healthz", headers={"X-Request-ID": "test-request-id"})
    assert resp.headers["X-Request-ID"] == "test-request-id"
