import pytest
from fastapi.testclient import TestClient

from backend import downloader, model_repository
from backend.app import app
from backend.utils import MODELS_DIR


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_index(client):
    assert client.get("/api/health").json()["app"] == "llm-autotuner"
    r = client.get("/")
    assert r.status_code == 200 and "LLM Autotuner" in r.text


def test_models_endpoint(client):
    rows = client.get("/api/models").json()
    assert len(rows) > 10
    assert {"id", "quant", "fit", "launch_ctx", "gpu_layers", "bench_estimated"} <= rows[0].keys()


def test_override_endpoint_changes_fit(client):
    hw = client.get("/api/hardware").json()
    hw["gpu"]["total_vram_gb"] = hw["gpu"]["free_vram_gb"] = 48
    big = {r["id"]: r for r in client.post("/api/models/override", json=hw).json()}
    assert big["unsloth/Qwen3-32B-GGUF"]["mode"] == "GPU"


def test_recommend_unknown_model(client):
    assert client.post("/api/recommend", json={"model_id": "nope/nope", "persona": "coding"}).status_code == 404


def test_rejects_foreign_host(client):
    # DNS-rebinding protection
    assert client.get("/api/health", headers={"host": "evil.example"}).status_code == 400


def test_rejects_cross_origin_post(client):
    r = client.post("/api/stop", headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    assert client.post("/api/stop", headers={"origin": "http://testserver"}).status_code == 200


def test_delete_rejects_traversal(client):
    assert client.delete("/api/installed/..%2F..%2Fmain.py").status_code in (400, 404)
    assert client.delete("/api/installed/../../main.py").status_code in (400, 404)


def test_launch_rejects_traversal_and_missing(client):
    r = client.post("/api/launch", json={"filename": "../../main.py"}).json()
    assert r["running"] is False
    r = client.post("/api/launch", json={"filename": "Nope-GGUF/nope.gguf"}).json()
    assert r["running"] is False and "download" in r["message"].lower()


def test_hf_files_rejects_bad_id(client):
    assert client.get("/api/hf-files/not a repo").status_code == 400


def test_installed_lists_shards_once_and_delete_cleans_folder(client):
    folder = MODELS_DIR / "Big-GGUF"
    folder.mkdir(parents=True, exist_ok=True)
    for i in (1, 2, 3):
        (folder / f"Big-Q4_K_M-0000{i}-of-00003.gguf").write_bytes(b"x" * 10)
    (folder / "mmproj-F16.gguf").write_bytes(b"x")
    items = [x for x in client.get("/api/installed").json() if x["filename"].startswith("Big-GGUF/")]
    assert len(items) == 1
    assert items[0]["shards"] == 3 and items[0]["complete"] and items[0]["has_mmproj"]
    assert items[0]["quant"] == "Q4_K_M" and items[0]["size_bytes"] == 30

    assert client.delete("/api/installed/" + items[0]["filename"]).status_code == 200
    assert not folder.exists()


def test_download_flow_with_stubbed_hf(client, monkeypatch):
    content = b"GGUF" + b"\0" * 2048
    tree = [{"path": "Tiny-Q4_K_M.gguf", "size": len(content)},
            {"path": "Tiny-Q8_0.gguf", "size": 999}]
    monkeypatch.setattr(model_repository, "repo_tree", lambda mid: tree)
    monkeypatch.setattr(downloader, "find_model", lambda mid: None)
    monkeypatch.setattr(downloader, "mmproj_file", lambda mid: None)

    class FakeStream:
        status_code = 200
        headers = {"content-length": str(len(content))}
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def aiter_bytes(self, n):
            yield content

    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def stream(self, method, url):
            assert url == "https://huggingface.co/unsloth/Tiny-GGUF/resolve/main/Tiny-Q4_K_M.gguf"
            return FakeStream()

    monkeypatch.setattr(downloader.httpx, "AsyncClient", FakeClient)

    groups = client.get("/api/hf-files/unsloth/Tiny-GGUF").json()
    assert sorted(g["quant"] for g in groups) == ["Q4_K_M", "Q8_0"]

    job = client.post("/api/download", json={"model_id": "unsloth/Tiny-GGUF", "quant": "Q4_K_M"}).json()
    assert job["filename"] == "Tiny-GGUF/Tiny-Q4_K_M.gguf"
    import time
    for _ in range(50):
        jobs = {j["key"]: j for j in client.get("/api/downloads").json()}
        if jobs[job["key"]]["status"] == "done":
            break
        time.sleep(0.05)
    assert jobs[job["key"]]["status"] == "done"
    assert (MODELS_DIR / "Tiny-GGUF" / "Tiny-Q4_K_M.gguf").read_bytes() == content
    inst = [x for x in client.get("/api/installed").json() if x["filename"] == job["filename"]]
    assert inst and inst[0]["model_id"] == "unsloth/Tiny-GGUF" and inst[0]["complete"]

    bad = client.post("/api/download", json={"model_id": "unsloth/Tiny-GGUF", "quant": "Q2_K"})
    assert bad.status_code == 400 and "available" in bad.json()["detail"]
    client.delete("/api/installed/" + job["filename"])


def test_runtime_status_shape(client):
    r = client.get("/api/runtime").json()
    assert {"installed", "installing", "variant"} <= r.keys()
