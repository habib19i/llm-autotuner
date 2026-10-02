"""Built-in chat relay (/api/chat)."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.launcher import LaunchStatus
from backend.utils import llm_api_key


@pytest.fixture
def client():
    with TestClient(app_module.app) as c:
        yield c


def _running(monkeypatch, ready=True):
    st = LaunchStatus(running=True, ready=ready, model="Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf", pid=1)
    monkeypatch.setattr(app_module, "launch_status", lambda: st)


def test_chat_needs_a_running_model(client):
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 409


def test_chat_waits_for_model_to_load(client, monkeypatch):
    _running(monkeypatch, ready=False)
    assert client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 409


def test_chat_relays_stream_with_api_key(client, monkeypatch):
    _running(monkeypatch)
    seen = {}

    def upstream(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        sse = (b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
               b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    monkeypatch.setattr(app_module, "_chat_transport", httpx.MockTransport(upstream))
    r = client.post("/api/chat", json={"messages": [{"role": "system", "content": "be brief"},
                                                    {"role": "user", "content": "hi"}],
                                       "temperature": 0.2})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert '"Hel"' in r.text and r.text.endswith("data: [DONE]\n\n")
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["auth"] == f"Bearer {llm_api_key()}"
    assert seen["body"]["stream"] is True and seen["body"]["temperature"] == 0.2
    assert seen["body"]["model"] == "Qwen3-8B-Q4_K_M.gguf"
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


def test_chat_image_parts_pass_through(client, monkeypatch):
    _running(monkeypatch)
    seen = {}

    def upstream(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    monkeypatch.setattr(app_module, "_chat_transport", httpx.MockTransport(upstream))
    parts = [{"type": "text", "text": "what is this?"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    assert client.post("/api/chat", json={"messages": [{"role": "user", "content": parts}]}).status_code == 200
    assert seen["body"]["messages"][0]["content"][1]["type"] == "image_url"


def test_chat_upstream_error_is_reported(client, monkeypatch):
    _running(monkeypatch)
    monkeypatch.setattr(app_module, "_chat_transport",
                        httpx.MockTransport(lambda r: httpx.Response(500, text="model crashed")))
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502 and "model crashed" in r.json()["detail"]


def test_chat_rejects_bad_roles(client, monkeypatch):
    _running(monkeypatch)
    r = client.post("/api/chat", json={"messages": [{"role": "tool", "content": "x"}]})
    assert r.status_code == 422
