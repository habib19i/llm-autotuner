"""
Benchmark the running model: real tokens/second and time to first token on this computer
(after llmfit's `bench`, MIT, github.com/AlexsJones/llmfit).

One short warm-up request, then three measured runs of a fixed prompt against the local
OpenAI-compatible server (llama.cpp or MLX). Results are kept in cache/benchmarks.json and
feed back into the catalog: a benchmarked model shows its measured speed, and runs on
this machine calibrate the estimates of every other model.
"""
import json
import statistics
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import httpx

from .utils import CACHE_DIR, LLM_PORT, get_logger, llm_api_key

logger = get_logger("bench")
STORE = CACHE_DIR / "benchmarks.json"
RUNS = 3
MAX_TOKENS = 160
PROMPT = ("Explain, in about 120 words, how a lighthouse keeper in the 1890s kept the lamp lit "
          "through a winter storm. Use plain language.")

_lock = threading.Lock()
_state: Dict[str, Any] = {"running": False, "stage": "idle", "run": 0, "runs": RUNS, "error": "", "result": None}


def fingerprint(hw) -> str:
    """Results only count on the machine that produced them."""
    return f"{hw.cpu_name}|{hw.gpu.name}|{round(hw.total_ram_gb)}GB"


def load() -> List[Dict[str, Any]]:
    try:
        data = json.loads(STORE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(results: List[Dict[str, Any]]):
    tmp = STORE.with_suffix(".tmp")
    tmp.write_text(json.dumps(results[-500:], indent=1), encoding="utf-8")
    tmp.replace(STORE)


def delete(filename: str) -> int:
    with _lock:
        results = load()
        keep = [r for r in results if r.get("filename") != filename]
        _save(keep)
        return len(results) - len(keep)


# ─── One request ──────────────────────────────────────────────────────────────

def measure_once(client: httpx.Client, base_url: str, model: str, max_tokens: int = MAX_TOKENS) -> Dict[str, float]:
    """Stream one completion; returns ttft_ms, tokens, tok_s (generation only) and prompt_tok_s."""
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    t_first = None
    chunks = 0
    usage_tokens = 0
    timings: Dict[str, Any] = {}
    with client.stream("POST", f"{base_url}/chat/completions", json=body) as r:
        if r.status_code != 200:
            r.read()
            raise RuntimeError(f"The model server answered {r.status_code}: {r.text[:200]}")
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            for ch in ev.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    chunks += 1
            if ev.get("usage"):
                usage_tokens = int(ev["usage"].get("completion_tokens") or 0)
            if ev.get("timings"):   # llama-server reports its own exact timings
                timings = ev["timings"]
    t_end = time.perf_counter()
    if t_first is None:
        raise RuntimeError("The model returned no text.")
    tokens = int(timings.get("predicted_n") or usage_tokens or chunks)
    if timings.get("predicted_per_second"):
        tok_s = float(timings["predicted_per_second"])
    else:
        gen_time = max(t_end - t_first, 1e-3)
        tok_s = max(tokens - 1, 1) / gen_time
    return {"ttft_ms": round((t_first - t0) * 1000, 1), "tokens": tokens, "tok_s": round(tok_s, 2),
            "prompt_tok_s": round(float(timings.get("prompt_per_second") or 0), 1)}


def run_benchmark(model: str, base_url: str, api_key: str, runs: int = RUNS,
                  client_factory: Optional[Callable[[], httpx.Client]] = None,
                  progress: Optional[Callable[[int], None]] = None) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    make = client_factory or (lambda: httpx.Client(timeout=httpx.Timeout(15.0, read=300.0), headers=headers))
    with make() as c:
        measure_once(c, base_url, model, max_tokens=8)   # warm-up: loads caches, not counted
        samples = []
        for i in range(runs):
            if progress:
                progress(i + 1)
            samples.append(measure_once(c, base_url, model))
    return {
        "tok_s": round(statistics.median(s["tok_s"] for s in samples), 1),
        "ttft_ms": round(statistics.median(s["ttft_ms"] for s in samples)),
        "prompt_tok_s": round(statistics.median(s["prompt_tok_s"] for s in samples), 1),
        "tokens": round(statistics.median(s["tokens"] for s in samples)),
        "runs": samples,
    }


# ─── Background job used by the API ───────────────────────────────────────────

def status() -> Dict[str, Any]:
    with _lock:
        return dict(_state)


def start(filename: str, model_id: str, quant: str, ctx: int, backend: str, hw,
          estimated_tok_s: Optional[float]) -> Dict[str, Any]:
    with _lock:
        if _state["running"]:
            return dict(_state)
        _state.update(running=True, stage="warming", run=0, error="", result=None)

    def progress(i: int):
        with _lock:
            _state.update(stage="measuring", run=i)

    def work():
        try:
            from pathlib import Path
            served = Path(filename).name   # llama-server's --alias; the MLX gateway accepts any name
            res = run_benchmark(served, f"http://127.0.0.1:{LLM_PORT}/v1", llm_api_key(), progress=progress)
            rec = {"filename": filename, "model_id": model_id, "quant": quant, "context": ctx,
                   "backend": backend, "hardware": fingerprint(hw), "estimated_tok_s": estimated_tok_s,
                   "date": time.strftime("%Y-%m-%d %H:%M"), **res}
            with _lock:
                results = load()
                results.append(rec)
                _save(results)
                _state.update(running=False, stage="done", result=rec)
            logger.info(f"Benchmark {served}: {res['tok_s']} tok/s, TTFT {res['ttft_ms']} ms")
        except Exception as e:
            logger.warning(f"Benchmark failed: {e}")
            with _lock:
                _state.update(running=False, stage="failed", error=str(e))

    threading.Thread(target=work, name="bench", daemon=True).start()
    return status()


# ─── Feeding results back into estimates ──────────────────────────────────────

def for_hardware(hw) -> List[Dict[str, Any]]:
    fp = fingerprint(hw)
    return [r for r in load() if r.get("hardware") == fp]


def latest_by_file(hw) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in for_hardware(hw):
        out[r["filename"]] = r   # later entries win
    return out


def calibration(hw) -> Optional[float]:
    """Median measured/estimated ratio of this machine's runs, or None without data."""
    ratios = [r["tok_s"] / r["estimated_tok_s"] for r in latest_by_file(hw).values()
              if r.get("estimated_tok_s") and r.get("tok_s")]
    if not ratios:
        return None
    return round(min(4.0, max(0.25, statistics.median(ratios))), 2)
