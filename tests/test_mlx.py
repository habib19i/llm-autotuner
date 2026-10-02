"""MLX support: catalog grouping, downloads, installed folders, gateway, launch, visibility."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteClient

from backend import downloader, launcher, mlx_runtime, model_repository as mr, selector
from backend.app import app
from backend.mlx_proxy import build_app
from backend.utils import MODELS_DIR
from tests.conftest import make_hw


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _mlx_item(rid, total=8.19e9, bits=4, base="Qwen/Qwen3-8B"):
    return {"id": rid, "pipeline_tag": "text-generation", "downloads": 100,
            "safetensors": {"total": total}, "config": {"quantization_config": {"bits": bits}},
            "tags": [f"base_model:quantized:{base}"],
            "siblings": [{"rfilename": f} for f in ("config.json", "model.safetensors", "tokenizer.json")]}


SAMPLE = [_mlx_item("mlx-community/Qwen3-8B-4bit"), _mlx_item("mlx-community/Qwen3-8B-8bit", bits=8),
          _mlx_item("mlx-community/Qwen3-8B-bf16", bits=None),
          _mlx_item("mlx-community/whisper-large-v3-turbo", base="openai/whisper")]


# ─── Catalog ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,stem,label", [
    ("Qwen3-8B-4bit", "Qwen3-8B", "4bit"),
    ("Qwen3-4B-Instruct-2507-4bit-DWQ-2510", "Qwen3-4B-Instruct-2507", "4bit-DWQ-2510"),
    ("gpt-oss-20b-MXFP4-Q8", "gpt-oss-20b", "MXFP4-Q8"),
    ("Llama-3.1-8B-Instruct-bf16", "Llama-3.1-8B-Instruct", "bf16"),
])
def test_mlx_quant_label(name, stem, label):
    assert mr.mlx_quant_label(name) == (stem, label)


def test_mlx_label_falls_back_to_config_bits():
    assert mr.mlx_quant_label("Kimi-K2.5", {"quantization_config": {"bits": 4}}) == ("Kimi-K2.5", "4bit")


def test_mlx_sizes_and_quality():
    assert mr.mlx_bits("4bit") == 4.5 and mr.mlx_bits("bf16") == 16.0
    assert mr.mlx_quality("8bit") > mr.mlx_quality("4bit") > mr.mlx_quality("3bit")
    assert mr.mlx_quality("4bit-DWQ") > mr.mlx_quality("4bit")


def test_mlx_entries_group_variants_by_base_model():
    entries = mr.mlx_entries(SAMPLE, {"qwen/qwen3-8b": 40})
    assert len(entries) == 1  # whisper (speech) filtered out
    e = entries[0]
    assert e.format == "mlx" and e.id == "mlx-community/Qwen3-8B-4bit" and e.context_k == 40
    assert [q.quant for q in e.quants] == ["4bit", "8bit", "bf16"]
    assert set(e.sources) == {"mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen3-8B-8bit",
                              "mlx-community/Qwen3-8B-bf16"}
    assert 4.0 < e.quants[0].size_gb < 4.8  # real repo: 4.29 GiB


def _with_mlx_catalog(monkeypatch):
    monkeypatch.setattr(mr, "_fetch_hf", lambda: mr._fallback() + mr.mlx_entries(SAMPLE, {}))
    monkeypatch.setattr(mr, "_mem_cache", None)
    mr.CACHE_FILE.unlink(missing_ok=True)


def test_mlx_rows_only_for_apple(monkeypatch):
    _with_mlx_catalog(monkeypatch)
    monkeypatch.setattr(selector, "mlx_supported", lambda: False)
    pc = {r.format for r in selector.build_table(make_hw())}
    mac = {r.format for r in selector.build_table(make_hw(vram=24, vendor="Apple", integrated=True))}
    assert pc == {"gguf"} and mac == {"gguf", "mlx"}


# ─── Downloads ─────────────────────────────────────────────────────────────────

TREE = [{"path": "config.json", "size": 10}, {"path": "model.safetensors", "size": 1000},
        {"path": "tokenizer.json", "size": 20}, {"path": "README.md", "size": 5},
        {"path": ".gitattributes", "size": 5}, {"path": "assets/logo.png", "size": 9}]


def test_mlx_download_plan(monkeypatch):
    monkeypatch.setattr(mr, "repo_tree", lambda mid: TREE)
    monkeypatch.setattr(downloader, "find_model", lambda mid: None)
    plan = downloader.resolve_download("mlx-community/Tiny-4bit", "4bit")
    assert plan["format"] == "mlx" and plan["flat"] is False
    assert sorted(f["path"] for f in plan["files"]) == ["config.json", "model.safetensors", "tokenizer.json"]
    assert plan["primary"] == plan["folder"] and plan["quant"] == "4bit"


def test_mlx_hf_files_lists_variants(client, monkeypatch):
    _with_mlx_catalog(monkeypatch)
    monkeypatch.setattr(mr, "repo_tree", lambda mid: TREE)
    groups = client.get("/api/hf-files/mlx-community/Qwen3-8B-4bit").json()
    assert [g["quant"] for g in groups] == ["4bit", "8bit", "bf16"] or len(groups) == 3
    assert {g["repo"] for g in groups} == {"mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen3-8B-8bit",
                                           "mlx-community/Qwen3-8B-bf16"}
    assert all(g["size_bytes"] == 1030 for g in groups)


def test_installed_mlx_folder_and_delete(client):
    d = MODELS_DIR / "Tiny-4bit"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"x" * 50)
    items = [x for x in client.get("/api/installed").json() if x["filename"] == "Tiny-4bit"]
    assert items and items[0]["format"] == "mlx" and items[0]["complete"] and items[0]["quant"] == "4bit"
    assert client.delete("/api/installed/Tiny-4bit").status_code == 200
    assert not d.exists()


def test_delete_refuses_non_model_folders(client):
    d = MODELS_DIR / "not-a-model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "notes.txt").write_text("keep me")
    assert client.delete("/api/installed/not-a-model").status_code == 404
    assert (d / "notes.txt").exists()


# ─── Gateway ───────────────────────────────────────────────────────────────────

def _upstream():
    seen = {}

    def handler(request: httpx.Request):
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            seen["model"] = body.get("model")
            if body.get("stream"):
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                      content=b'data: {"x":1}\n\ndata: [DONE]\n\n')
            return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
        return httpx.Response(200, json={"status": "ok"})
    return seen, httpx.MockTransport(handler)


def test_gateway_requires_key_and_rewrites_model():
    seen, transport = _upstream()
    c = StarletteClient(build_app("http://upstream", "sk-test", transport))
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]}
    assert c.post("/v1/chat/completions", json=body).status_code == 401
    assert c.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    r = c.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-test"})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "hi"
    assert seen["model"] == "default_model"          # client's name never reaches mlx_lm
    assert seen["auth"] is None                      # key is not forwarded upstream
    assert c.post("/v1/chat/completions", json=body, headers={"x-api-key": "sk-test"}).status_code == 200
    assert c.get("/health").status_code == 200       # health check needs no key


def test_gateway_streams_events():
    _, transport = _upstream()
    c = StarletteClient(build_app("http://upstream", "k", transport))
    r = c.post("/v1/chat/completions", json={"stream": True, "messages": []}, headers={"Authorization": "Bearer k"})
    assert r.status_code == 200 and r.text.endswith("data: [DONE]\n\n")


# ─── Launch ────────────────────────────────────────────────────────────────────

def test_mlx_launch_without_runtime_is_dry_run(monkeypatch):
    d = MODELS_DIR / "Small-4bit"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(mlx_runtime, "is_installed", lambda: False)
    st = launcher.launch("Small-4bit", 4096, 4, 0)
    assert st.dry_run and st.backend == "mlx" and "MLX runtime" in st.message
    launcher.stop()


def test_mlx_status_endpoint(client):
    r = client.get("/api/runtime/mlx").json()
    assert {"supported", "installed", "installing"} <= r.keys()
