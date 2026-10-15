import json

import pytest
from fastapi.testclient import TestClient

from backend import model_repository, opencode
from backend.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def cfg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path / "opencode"


MODEL = "unsloth_Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
KEY = "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"


def test_strip_jsonc_keeps_strings():
    text = '{\n // comment\n "a": "http://x//y", /* block */ "b": [1, 2,],\n}'
    assert json.loads(opencode.strip_jsonc(text)) == {"a": "http://x//y", "b": [1, 2]}
    assert json.loads(opencode.strip_jsonc('{"s": "a,}", "t": [1, // c\n],}')) == {"s": "a,}", "t": [1]}


def test_configure_creates_config(cfg_home):
    st = opencode.configure(MODEL, 32768, "Qwen3 Coder")
    data = json.loads((cfg_home / "opencode.json").read_text())
    prov = data["provider"]["llm-autotuner"]
    assert prov["npm"] == "@ai-sdk/openai-compatible"
    assert prov["options"]["baseURL"].startswith("http://127.0.0.1:") and prov["options"]["baseURL"].endswith("/v1")
    assert prov["options"]["apiKey"]
    assert list(prov["models"]) == [KEY]
    assert prov["models"][KEY]["limit"]["context"] == 32768
    assert data["model"] == f"llm-autotuner/{KEY}"
    assert st["configured"] and st["is_default"] and st["model"] == KEY


def test_configure_keeps_other_settings_and_backs_up(cfg_home):
    cfg_home.mkdir(parents=True)
    original = {"theme": "tokyonight", "model": "anthropic/claude", "provider": {"other": {"npm": "x"}}}
    (cfg_home / "opencode.json").write_text(json.dumps(original))
    opencode.configure(MODEL, 16384, set_default=False)
    data = json.loads((cfg_home / "opencode.json").read_text())
    assert data["theme"] == "tokyonight" and data["model"] == "anthropic/claude"
    assert set(data["provider"]) == {"other", "llm-autotuner"}
    assert json.loads((cfg_home / "opencode.json.bak").read_text()) == original


def test_configure_replaces_previous_model(cfg_home):
    opencode.configure(MODEL, 16384)
    opencode.configure("bartowski_Qwen_Qwen3-8B-GGUF/Qwen_Qwen3-8B-Q4_K_M.gguf", 16384)
    data = json.loads((cfg_home / "opencode.json").read_text())
    assert list(data["provider"]["llm-autotuner"]["models"]) == ["Qwen_Qwen3-8B-Q4_K_M.gguf"]


def test_jsonc_config_is_edited_in_place(cfg_home):
    cfg_home.mkdir(parents=True)
    (cfg_home / "opencode.jsonc").write_text('{\n  // mine\n  "theme": "opencode",\n}')
    opencode.configure(MODEL, 16384)
    assert not (cfg_home / "opencode.json").exists()
    assert json.loads((cfg_home / "opencode.jsonc").read_text())["theme"] == "opencode"


def test_invalid_config_is_not_overwritten(cfg_home, client):
    cfg_home.mkdir(parents=True)
    (cfg_home / "opencode.json").write_text("{ not json")
    r = client.post("/api/opencode/configure", json={"filename": MODEL, "ctx": 16384})
    assert r.status_code == 400
    assert (cfg_home / "opencode.json").read_text() == "{ not json"


def test_configure_rejects_traversal(cfg_home, client):
    r = client.post("/api/opencode/configure", json={"filename": "../../etc/passwd", "ctx": 16384})
    assert r.status_code == 400
    assert not (cfg_home / "opencode.json").exists()


def test_api_status_and_preview(cfg_home, client):
    st = client.get("/api/opencode").json()
    assert st["configured"] is False and st["install"]
    block = client.post("/api/opencode/preview", json={"filename": MODEL, "ctx": 65536}).json()
    assert block["model"] == f"llm-autotuner/{KEY}"
    assert not (cfg_home / "opencode.json").exists()


def test_regenerating_key_updates_opencode(cfg_home, client):
    opencode.configure(MODEL, 16384)
    old = json.loads((cfg_home / "opencode.json").read_text())["provider"]["llm-autotuner"]["options"]["apiKey"]
    new = client.post("/api/settings/api-key/regenerate").json()["api_key"]
    cur = json.loads((cfg_home / "opencode.json").read_text())["provider"]["llm-autotuner"]["options"]["apiKey"]
    assert cur == new != old


def test_open_terminal_without_opencode(cfg_home, client, monkeypatch):
    monkeypatch.setattr(opencode, "find_exe", lambda: None)
    assert client.post("/api/opencode/open", json={}).status_code == 404


def test_open_terminal_rejects_missing_folder(cfg_home, client, monkeypatch, tmp_path):
    monkeypatch.setattr(opencode, "find_exe", lambda: "/usr/bin/opencode")
    r = client.post("/api/opencode/open", json={"folder": str(tmp_path / "nope")})
    assert r.status_code == 400


@pytest.mark.parametrize("mlx", [False, True])
def test_search_hf(client, monkeypatch, mlx):
    from backend import mlx_runtime
    monkeypatch.setattr(mlx_runtime, "mlx_supported", lambda: mlx)
    calls = []

    def fake(params):
        calls.append(dict(params))
        if dict(params)["filter"] == "mlx":  # macOS runs an MLX search too
            return [{"id": "unsloth/Qwen3-8B-GGUF", "downloads": 900},
                    {"id": "mlx-community/Cool-Model-12B-4bit", "downloads": 10, "safetensors": {"total": 12.1e9}}]
        return [
            {"id": "unsloth/Qwen3-8B-GGUF", "downloads": 900, "gguf": {"total": 8.19e9, "context_length": 40960}},
            {"id": "someone/Cool-Model-12B-GGUF", "downloads": 50, "likes": 3, "gguf": {"total": 12.1e9}},
            {"id": "someone/Cool-Embedding-GGUF", "downloads": 70},
            {"id": "someone/Img-GGUF", "pipeline_tag": "text-to-image"},
        ]
    monkeypatch.setattr(model_repository, "_hf_search_raw", fake)
    res = client.get("/api/models/search", params={"q": "cool"}).json()
    assert calls[0]["filter"] == "gguf" and calls[0]["search"] == "cool"
    ids = [r["id"] for r in res if r["format"] == "gguf"]
    assert ids == ["unsloth/Qwen3-8B-GGUF", "someone/Cool-Model-12B-GGUF"]
    assert len({r["id"] for r in res}) == len(res)  # no duplicates across the two searches
    assert res[0]["in_catalog"] == "unsloth/Qwen3-8B-GGUF" and res[0]["context_k"] == 40
    assert res[1]["in_catalog"] == "" and res[1]["params_b"] == 12.1
    assert client.get("/api/models/search", params={"q": "a"}).json() == []


def test_new_publishers_parse():
    items = [
        {"id": "mradermacher/Hermes-4-14B-i1-GGUF", "downloads": 9000,
         "gguf": {"total": 14.8e9, "context_length": 40960, "architecture": "qwen3"},
         "tags": ["base_model:quantized:NousResearch/Hermes-4-14B"],
         "siblings": [{"rfilename": "Hermes-4-14B.i1-Q4_K_M.gguf"}, {"rfilename": "Hermes-4-14B.i1-Q6_K.gguf"}]},
        {"id": "google/gemma-3-4b-it-qat-q4_0-gguf", "downloads": 5000,
         "gguf": {"total": 3.88e9, "context_length": 131072, "architecture": "gemma3"},
         "siblings": [{"rfilename": "gemma-3-4b-it-q4_0.gguf"}, {"rfilename": "mmproj-model-f16-4B.gguf"}]},
    ]
    a, b = model_repository._from_hf(items)
    assert a.name == "Hermes 4 14B" and a.provider == "Nous Research"
    assert {q.quant for q in a.quants} == {"Q4_K_M", "Q6_K"}
    assert b.provider == "Google" and b.is_vision and [q.quant for q in b.quants] == ["Q4_0"]
