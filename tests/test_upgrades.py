"""Catalog sources, custom models, quality data, runtime selection, disk checks, settings."""
import pytest
from fastapi.testclient import TestClient

from backend import benchmark_provider as bp, downloader, model_repository as mr, runtime, launcher, utils
from backend.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ─── Catalog sources ───────────────────────────────────────────────────────────

def _item(rid, base, files=("X-Q4_K_M.gguf",), total=8e9, gated=False):
    return {"id": rid, "pipeline_tag": "text-generation", "gated": gated,
            "gguf": {"total": total, "architecture": "qwen3", "context_length": 40960},
            "tags": [f"base_model:{base}", f"base_model:quantized:{base}"],
            "siblings": [{"rfilename": f} for f in files]}


def test_merge_sources_dedupes_by_base_model_in_priority_order():
    a = mr._from_hf([_item("unsloth/Qwen3-8B-GGUF", "Qwen/Qwen3-8B")])
    b = mr._from_hf([_item("bartowski/Qwen_Qwen3-8B-GGUF", "Qwen/Qwen3-8B")])
    c = mr._from_hf([_item("bartowski/Other-7B-GGUF", "Org/Other-7B", gated=True)])
    merged = mr.merge_sources(a + b + c)
    assert len(merged) == 2
    q = next(m for m in merged if m.base_model == "Qwen/Qwen3-8B")
    assert q.id == "unsloth/Qwen3-8B-GGUF"
    assert q.sources == ["unsloth/Qwen3-8B-GGUF", "bartowski/Qwen_Qwen3-8B-GGUF"]
    assert next(m for m in merged if m.id.endswith("Other-7B-GGUF")).gated


def test_bartowski_org_prefix_stripped_from_name():
    m = mr._from_hf([_item("bartowski/Qwen_Qwen3-8B-GGUF", "Qwen/Qwen3-8B")])[0]
    assert m.name == "Qwen3 8B" and m.publisher == "bartowski" and m.provider == "Alibaba"


def test_non_chat_repos_filtered_by_name():
    assert mr._from_hf([_item("ggml-org/Qwen3-TTS-1.7B-GGUF", "Qwen/Qwen3-TTS")]) == []


@pytest.mark.parametrize("text,rid", [
    ("https://huggingface.co/Qwen/Qwen3-8B-GGUF", "Qwen/Qwen3-8B-GGUF"),
    ("https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/main/x.gguf", "Qwen/Qwen3-8B-GGUF"),
    ("hf.co/bartowski/Foo-GGUF?x=1", "bartowski/Foo-GGUF"),
    ("  unsloth/Qwen3-8B-GGUF ", "unsloth/Qwen3-8B-GGUF"),
])
def test_parse_repo_id(text, rid):
    assert mr.parse_repo_id(text) == rid


@pytest.mark.parametrize("bad", ["", "not a link", "https://huggingface.co/datasets/foo/bar", "https://evil.com/a/b"])
def test_parse_repo_id_rejects(bad):
    with pytest.raises(ValueError):
        mr.parse_repo_id(bad)


def test_custom_model_add_and_remove(client, monkeypatch):
    class R:
        status_code = 200
        def json(self): return _item("someone/Tiny-GGUF", "someone/Tiny", files=("Tiny-Q8_0.gguf",), total=1e9)
        def raise_for_status(self): pass

    class C:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url, params=None): return R()

    monkeypatch.setattr(mr.httpx, "Client", C)
    r = client.post("/api/models/custom", json={"url": "https://huggingface.co/someone/Tiny-GGUF"})
    assert r.status_code == 200 and r.json()["id"] == "someone/Tiny-GGUF"
    rows = {x["id"]: x for x in client.get("/api/models").json()}
    assert rows["someone/Tiny-GGUF"]["custom"] is True
    assert client.delete("/api/models/custom/someone/Tiny-GGUF").status_code == 200
    assert "someone/Tiny-GGUF" not in {x["id"] for x in client.get("/api/models").json()}
    assert client.post("/api/models/custom", json={"url": "nonsense"}).status_code == 400


# ─── Quality data ──────────────────────────────────────────────────────────────

def test_arena_name_matching():
    idx = bp._index({"llama-3.1-8b-instruct": 1186.0, "gemma-4-31b": 1443.0,
                     "nvidia-nemotron-3-nano-30b-a3b-bf16": 1348.0})
    assert bp.match_rating(idx, "meta-llama/Llama-3.1-8B-Instruct") == 1186.0
    assert bp.match_rating(idx, "google/gemma-4-31B-it") == 1443.0
    assert bp.match_rating(idx, "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16") == 1348.0
    assert bp.match_rating(idx, "Qwen/Qwen3-8B") is None


def test_catalog_benchmarks_measured(monkeypatch):
    models = mr.get_models()
    names = {"llama-3.1-8b-instruct": 1186.0}
    monkeypatch.setattr(bp, "get_arena", lambda refresh=True: {"ts": 1, "published": "2026-09-30",
                                                             "text": names, "webdev": {}, "vision": {}})
    monkeypatch.setattr(bp, "_cache_key", None)
    b = bp.catalog_benchmarks(models)
    llama = b["unsloth/Llama-3.1-8B-Instruct-GGUF"]
    assert llama.arena_elo == 1186 and not llama.estimated and "LMArena" in llama.source
    assert llama.mmlu is not None  # published numbers still shown for known models
    other = b["unsloth/Qwen3-8B-GGUF"]
    assert other.estimated and other.arena_elo is None and other.mmlu is None


def test_estimates_are_flagged_and_bounded(monkeypatch):
    monkeypatch.setattr(bp, "get_arena", lambda refresh=True: {"ts": 2, "published": "", "text": {},
                                                             "webdev": {}, "vision": {}})
    monkeypatch.setattr(bp, "_cache_key", None)
    for b in bp.catalog_benchmarks(mr.get_models()).values():
        assert b.estimated and b.arena_elo is None and 950 <= b.rating <= 1450


def test_fit_recovers_size_slope():
    import math
    pts = [(p, 2025.0, 1100 + 50 * math.log(p)) for p in (1, 2, 4, 8, 16, 32, 64, 3, 6, 12, 24, 48)]
    a, b, c = bp._fit(pts)
    assert abs(b - 50) < 1 and abs(a - 1100) < 1


def test_rows_expose_quality_fields(client):
    row = client.get("/api/models").json()[0]
    assert {"rating", "arena_elo", "bench_source", "sources", "publisher", "gated"} <= row.keys()


# ─── Runtime selection ─────────────────────────────────────────────────────────

RELS = [{"assets": [{"name": n, "browser_download_url": "https://x/" + n, "size": 1} for n in (
    "llama-b200-bin-win-cuda-12.4-x64.zip", "llama-b200-bin-win-cuda-13.4-x64.zip",
    "cudart-llama-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip",
    "llama-b200-bin-win-vulkan-x64.zip", "llama-b200-bin-win-cpu-x64.zip",
    "llama-b200-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b200-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "llama-b200-bin-ubuntu-vulkan-x64.tar.gz", "llama-b200-bin-macos-arm64.tar.gz")]}]


@pytest.mark.parametrize("vendor,driver,system,expect,extra", [
    ("NVIDIA", "581.15", "Windows", "win-cuda-13.4-x64", True),
    ("NVIDIA", "560.94", "Windows", "win-cuda-12.4-x64", True),
    ("NVIDIA", "472.12", "Windows", "win-vulkan-x64", False),
    ("AMD", "", "Windows", "win-vulkan-x64", False),
    ("", "", "Windows", "win-cpu-x64", False),
    ("NVIDIA", "570.10", "Linux", "ubuntu-cuda-12.8-x64", True),
    ("", "", "Darwin", "macos-arm64", False),
])
def test_runtime_variant_selection(monkeypatch, vendor, driver, system, expect, extra):
    machine = "arm64" if system == "Darwin" else "x86_64"
    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    monkeypatch.setattr(runtime.platform, "machine", lambda: machine)
    a = runtime.best_asset(vendor, driver, RELS)
    assert a["variant"] == expect and a["tag"] == "b200"
    assert bool(a["extra"]) == extra


def test_runtime_prefer_vulkan_over_cuda(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "AMD64")
    assert runtime.best_asset("NVIDIA", "581.15", RELS, prefer="win-vulkan-x64")["variant"] == "win-vulkan-x64"


# ─── Disk space ────────────────────────────────────────────────────────────────

def test_download_refused_when_disk_full(client, monkeypatch):
    monkeypatch.setattr(mr, "repo_tree", lambda mid: [{"path": "Big-Q4_K_M.gguf", "size": 50 * 1024 ** 3}])
    monkeypatch.setattr(downloader, "find_model", lambda mid: None)
    monkeypatch.setattr(downloader, "mmproj_file", lambda mid: None)

    class U:
        free = 10 * 1024 ** 3
        total = 100 * 1024 ** 3
    monkeypatch.setattr(downloader.shutil, "disk_usage", lambda p: U)
    r = client.post("/api/download", json={"model_id": "someone/Big-GGUF", "quant": "Q4_K_M"})
    assert r.status_code == 507 and "disk space" in r.json()["detail"]


def test_disk_endpoint(client):
    d = client.get("/api/disk").json()
    assert d["free_bytes"] > 0 and "models_gb" in d


# ─── Settings / API key ────────────────────────────────────────────────────────

def test_settings_and_key_regeneration(client):
    s1 = client.get("/api/settings").json()
    assert s1["api_key"].startswith("sk-local-") and not s1["hf_token_set"]
    s2 = client.post("/api/settings/api-key/regenerate").json()
    assert s2["api_key"] != s1["api_key"]


def test_bad_hf_token_rejected(client, monkeypatch):
    class R:
        status_code = 401
    monkeypatch.setattr("backend.app.httpx.get", lambda *a, **k: R())
    assert client.post("/api/settings/hf-token", json={"token": "hf_invalid_token"}).status_code == 400
    assert not client.get("/api/settings").json()["hf_token_set"]


def test_hf_headers_use_saved_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    utils.set_setting("hf_token", "hf_abc123456")
    try:
        assert utils.hf_headers() == {"Authorization": "Bearer hf_abc123456"}
    finally:
        utils.set_setting("hf_token", None)


def test_launch_passes_api_key_but_hides_it(monkeypatch):
    folder = utils.MODELS_DIR / "K-GGUF"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "K-Q4_K_M.gguf").write_bytes(b"x")
    monkeypatch.setattr(launcher, "find_server", lambda: None)  # dry run: returns the command
    st = launcher.launch("K-GGUF/K-Q4_K_M.gguf", 2048, 2, 0)
    key = utils.llm_api_key()
    assert "--api-key ***" in st.cmd and key not in st.cmd
    launcher.stop()


def test_finetunes_do_not_inherit_published_scores():
    assert bp._published("meta-llama/Llama-3.1-8B-Instruct") is not None
    assert bp._published("Llama-3.1-8B-Stheno-v3.4") is None
    assert bp._published("google/gemma-2-9b-it") is not None
