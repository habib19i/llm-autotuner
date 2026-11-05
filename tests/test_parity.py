"""Features matching llmfit: hardware profiles, plan, storage, benchmarks, other runtimes, /api/v1."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import bench, other_runtimes, profiles, selector
from backend.app import app
from backend.model_repository import find_model
from backend.storage_plan import storage_plan
from backend.utils import set_setting
from tests.conftest import make_hw


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clean_state():
    bench.STORE.unlink(missing_ok=True)
    set_setting("speed_efficiency", None)
    set_setting("max_context", None)
    yield
    bench.STORE.unlink(missing_ok=True)
    set_setting("speed_efficiency", None)
    set_setting("max_context", None)


# ─── Hardware profiles ─────────────────────────────────────────────────────────

def test_gpu_bandwidth_prefers_longest_name():
    assert profiles.gpu_bandwidth("NVIDIA GeForce RTX 4070 Ti SUPER") == 672
    assert profiles.gpu_bandwidth("NVIDIA GeForce RTX 4070") == 504
    assert profiles.gpu_bandwidth("Apple M2 Pro GPU") == 200
    assert profiles.gpu_bandwidth("NVIDIA RTX A1000") is None   # not "A100"
    assert profiles.gpu_bandwidth("Mystery GPU") is None


def test_known_gpu_changes_speed_estimate():
    m = find_model("unsloth/Qwen3-8B-GGUF")
    q = next(x for x in m.quants if x.quant == "Q4_K_M")
    slow, fast = make_hw(vram=24), make_hw(vram=24)
    slow.gpu.name, fast.gpu.name = "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 5090"
    assert selector.estimate(m, q, fast).tok_s > selector.estimate(m, q, slow).tok_s


def test_profiles_endpoint_and_simulation(client):
    lst = client.get("/api/hardware/profiles").json()
    assert {"rtx-4090-24gb", "apple-m4-max-128gb"} <= {p["id"] for p in lst}
    hw = client.get("/api/hardware/profiles/rtx-4090-24gb").json()
    assert hw["gpu"]["total_vram_gb"] == 24 and hw["gpu_bandwidth_gbps"] == 1008 and hw["simulated"]
    rows = {r["id"]: r for r in client.post("/api/models/override", json=hw).json()}
    assert rows["unsloth/Qwen3-14B-GGUF"]["mode"] == "GPU"
    assert client.get("/api/hardware/profiles/nope").status_code == 404


# ─── Plan ──────────────────────────────────────────────────────────────────────

def test_plan_for_model_too_big_for_gpu(client):
    p = client.post("/api/plan", json={"model_id": "unsloth/Qwen3-32B-GGUF", "ctx": 16384,
                                       "target_tps": 20}).json()
    assert p["quant"] == "Q4_K_M" and p["total_gb"] > 18
    paths = {x["path"]: x for x in p["paths"]}
    assert paths["GPU"]["fit_here"] == "No Fit" and p["upgrade"]["memory_gb"] > 0
    assert paths["GPU"]["min_vram_gb"] == p["total_gb"] and paths["CPU"]["min_vram_gb"] == 0
    assert paths["GPU"]["tok_s"] > paths["CPU"]["tok_s"]
    assert p["needed_bandwidth_gbps"] > 0
    fits = [m for m in p["machines"] if m["fit"] != "No Fit"]
    assert fits and fits[0]["id"] in {x["id"] for x in profiles.PROFILES}


def test_plan_small_model_and_sessions(client):
    p = client.post("/api/plan", json={"model_id": "unsloth/Qwen3-1.7B-GGUF", "ctx": 4096, "quant": "Q8_0"}).json()
    assert p["quant"] == "Q8_0" and p["upgrade"] is None and p["parallel_sessions"] >= 2
    assert p["paths"][0]["fit_here"] in ("Perfect", "Good")


def test_plan_errors(client):
    assert client.post("/api/plan", json={"model_id": "no/such"}).status_code == 404
    r = client.get("/api/v1/plan", params={"model": "unsloth/Qwen3-8B-GGUF", "context": 8192, "quant": "Q5_K_M"})
    assert r.status_code == 200 and r.json()["context"] == 8192


# ─── Storage ───────────────────────────────────────────────────────────────────

def test_storage_plan_math():
    rows = selector.build_table(make_hw(vram=24, ram=64, avail=56))
    st = storage_plan(rows, keep=2, os_reserve_gb=100, headroom_pct=15)
    assert len(st["models"]) == 2
    lib = round(sum(m["disk_gb"] for m in st["models"]), 1)
    assert st["library_gb"] == lib
    assert st["download_scratch_gb"] == max(m["disk_gb"] for m in st["models"])
    assert st["need_gb"] == round(100 + lib + st["download_scratch_gb"], 1)
    assert st["minimum_ssd_gb"] in (256, 512) and st["suggested_ssd_gb"] >= st["minimum_ssd_gb"]
    big = storage_plan(rows, keep=3, selection="largest", os_reserve_gb=0, scratch_gb=0, headroom_pct=0)
    assert [m["disk_gb"] for m in big["models"]] == sorted((m["disk_gb"] for m in big["models"]), reverse=True)
    with pytest.raises(ValueError):
        storage_plan(rows, keep=0)


def test_storage_endpoints(client):
    st = client.post("/api/storage-plan", json={"keep": 1, "search": "qwen"}).json()
    assert len(st["models"]) == 1 and "qwen" in st["models"][0]["name"].lower()
    v1 = client.get("/api/v1/storage", params={"keep": 2, "os_reserve": "1T", "scratch": "0"}).json()["storage"]
    assert v1["os_reserve_gb"] == 1000 and v1["download_scratch_gb"] == 0
    assert client.get("/api/v1/storage", params={"os_reserve": "lots"}).status_code == 400


# ─── Benchmarks ────────────────────────────────────────────────────────────────

def _sse(events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def _fake_server(with_timings: bool):
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        n = body["max_tokens"]
        evs = [{"choices": [{"delta": {"content": f"w{i} "}}]} for i in range(min(n, 20))]
        last = {"choices": [], "usage": {"completion_tokens": min(n, 20)}}
        if with_timings:
            last["timings"] = {"predicted_n": min(n, 20), "predicted_per_second": 42.5, "prompt_per_second": 300.0}
        evs.append(last)
        return httpx.Response(200, text=_sse(evs), headers={"content-type": "text/event-stream"})
    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


def test_benchmark_uses_server_timings():
    res = bench.run_benchmark("m.gguf", "http://x/v1", "", runs=3, client_factory=_fake_server(True))
    assert res["tok_s"] == 42.5 and res["prompt_tok_s"] == 300.0 and res["tokens"] == 20
    assert len(res["runs"]) == 3 and res["ttft_ms"] >= 0


def test_benchmark_without_timings_counts_tokens():
    res = bench.run_benchmark("m.gguf", "http://x/v1", "", runs=2, client_factory=_fake_server(False))
    assert res["tok_s"] > 0 and res["tokens"] == 20


def test_benchmark_reports_server_error():
    bad = lambda: httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="no key")))
    with pytest.raises(RuntimeError, match="401"):
        bench.run_benchmark("m", "http://x/v1", "", client_factory=bad)


def test_benchmarks_feed_back_into_table():
    hw = make_hw()
    rows = {r.id: r for r in selector.build_table(hw)}
    row = rows["unsloth/Qwen3-8B-GGUF"]
    other = rows["unsloth/Qwen3-4B-Instruct-2507-GGUF"]
    est_other = other.tok_s
    bench._save([{"filename": "unsloth_Qwen3-8B-GGUF/x.gguf", "model_id": row.id, "quant": row.quant,
                  "hardware": bench.fingerprint(hw), "estimated_tok_s": row.tok_s, "tok_s": row.tok_s * 2,
                  "ttft_ms": 120, "date": "2026-11-05 10:00"}])
    assert bench.calibration(hw) == 2.0
    rows = {r.id: r for r in selector.build_table(hw)}
    assert rows[row.id].tok_s_basis == "measured" and rows[row.id].measured_ttft_ms == 120
    assert rows[other.id].tok_s_basis == "calibrated" and rows[other.id].tok_s == pytest.approx(est_other * 2, 0.02)
    # Another machine's results don't count, and simulated hardware is never calibrated
    sim = make_hw()
    sim.simulated = "Custom"
    assert all(r.tok_s_basis == "estimate" for r in selector.build_table(sim))
    other_hw = make_hw()
    other_hw.cpu_name = "Other CPU"
    assert bench.calibration(other_hw) is None


def test_bench_requires_running_model(client):
    assert client.post("/api/bench").status_code == 409
    assert client.get("/api/bench").json()["running"] is False
    assert client.get("/api/v1/benchmarks").json()["results"] == []


# ─── Other runtimes ────────────────────────────────────────────────────────────

def test_match_keys():
    k = other_runtimes.match_key
    assert k("qwen3:8b") == k("Qwen3 8B") == k("lmstudio-community/Qwen3-8B-GGUF") == "qwen38b"
    assert k("llama3.2:3b") == k("Llama 3.2 3B Instruct")
    assert k("gemma3:4b") == k("gemma 3 4b it")
    assert k("qwen3:8b-q4_K_M") == "qwen38b"


def test_other_runtime_matches(client, monkeypatch):
    found = {"ollama": {"name": "Ollama", "url": "u", "running": True, "models": ["qwen3:8b", "llama3.2:latest"]},
             "lmstudio": {"name": "LM Studio", "url": "u", "running": True, "models": ["google/gemma-3-4b"]}}
    monkeypatch.setattr(other_runtimes, "detect", lambda force=False: found)
    res = client.get("/api/runtimes/others").json()
    assert res["matches"]["unsloth/Qwen3-8B-GGUF"] == ["Ollama: qwen3:8b"]
    assert res["matches"]["unsloth/gemma-3-4b-it-GGUF"] == ["LM Studio: google/gemma-3-4b"]
    assert not any("latest" in h for hits in res["matches"].values() for h in hits)


# ─── Settings, diagnostics, /api/v1 ────────────────────────────────────────────

def test_estimate_settings(client):
    base = {r["id"]: r for r in client.get("/api/models").json()}
    st = client.post("/api/settings/estimates", json={"speed_efficiency": 0.3, "max_context": 4096}).json()
    assert st["speed_efficiency"] == 0.3 and st["max_context"] == 4096
    slow = {r["id"]: r for r in client.get("/api/models").json()}
    rid = "unsloth/Qwen3-8B-GGUF"
    assert slow[rid]["tok_s"] < base[rid]["tok_s"] and slow[rid]["launch_ctx"] <= 4096
    assert all(r["agent_ctx"] == 0 for r in slow.values())   # capped below agent sizes
    assert client.post("/api/settings/estimates", json={"speed_efficiency": 5}).status_code == 422


def test_doctor(client):
    r = client.get("/api/doctor")
    assert r.status_code == 200 and "Detected hardware" in r.text and "hf_" not in r.text


def test_v1_endpoints(client):
    sysinfo = client.get("/api/v1/system").json()
    assert sysinfo["hardware"]["cpu_name"] == "Test CPU" and sysinfo["gpu_bandwidth_gbps"] > 0
    allm = client.get("/api/v1/models").json()
    assert allm["count"] == len(allm["models"]) > 10
    perfect = client.get("/api/v1/models", params={"min_fit": "perfect", "sort": "tps", "limit": 3}).json()
    assert len(perfect["models"]) <= 3 and all(m["fit"] == "Perfect" for m in perfect["models"])
    tps = [m["tok_s"] for m in perfect["models"]]
    assert tps == sorted(tps, reverse=True)
    coding = client.get("/api/v1/models", params={"use_case": "coding"}).json()["models"]
    assert coding and all(m["is_coding"] for m in coding)
    top = client.get("/api/v1/models/top", params={"limit": 2}).json()
    assert top["count"] <= 2 and all(m["fit"] in ("Perfect", "Good") for m in top["models"])
    one = client.get("/api/v1/models/unsloth/Qwen3-8B-GGUF").json()
    assert one["count"] == 1 and one["models"][0]["id"] == "unsloth/Qwen3-8B-GGUF"
    assert client.get("/api/v1/models/zzz-nothing").status_code == 404
    assert client.get("/api/v1/models", params={"min_fit": "bad"}).status_code == 400
    assert client.get("/api/v1/models", params={"sort": "bad"}).status_code == 400
    big = client.get("/api/v1/models/top", params={"vram_gb": 80, "ram_gb": 256}).json()
    assert big["simulated"] == "Custom"
    prof = client.get("/api/v1/models/top", params={"profile": "apple-m4-max-128gb"}).json()
    assert prof["simulated"].startswith("MacBook Pro")
    assert client.get("/api/v1/system", params={"profile": "nope"}).status_code == 404
