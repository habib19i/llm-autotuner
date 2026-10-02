"""
Model quality data.

- LMArena ratings (live, open-weight models) are the primary quality signal.
- Well-known older models also carry their published benchmark results.
- Models that are on neither get a rating estimated from size and release date, calibrated
  against the measured models, and are flagged `estimated` so the UI can mark them with ~.
"""
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel

from .utils import CACHE_DIR, get_logger

logger = get_logger("benchmarks")


class Benchmarks(BaseModel):
    # Published results (None when not published for this model)
    swe_bench: Optional[float] = None   # software engineering agent score
    humaneval: Optional[float] = None   # Python code pass@1
    gpqa: Optional[float] = None        # graduate-level science
    ifeval: Optional[float] = None      # strict instruction following
    mmlu: Optional[float] = None        # general knowledge
    # LMArena
    rating: float = 1200.0              # measured arena rating, or estimate when `estimated`
    arena_elo: Optional[int] = None     # measured text-arena rating
    arena_coding: Optional[int] = None  # measured WebDev-arena rating
    arena_vision: Optional[int] = None  # measured vision-arena rating
    estimated: bool = True
    source: str = ""


# Key: lowercase fragment of model name — first match wins
_DB: Dict[str, Dict] = {
    "qwen2.5-coder-7b":   {"swe_bench": 41.5, "humaneval": 84.1, "gpqa": 38.0, "ifeval": 76.5, "mmlu": 74.2, "arena_elo": 1235},
    "qwen2.5-coder-14b":  {"swe_bench": 46.0, "humaneval": 87.2, "gpqa": 43.0, "ifeval": 80.0, "mmlu": 78.5, "arena_elo": 1265},
    "qwen2.5-coder-32b":  {"swe_bench": 52.3, "humaneval": 90.2, "gpqa": 47.5, "ifeval": 84.0, "mmlu": 83.0, "arena_elo": 1300},
    "qwen3-coder":        {"swe_bench": 61.0, "humaneval": 93.5, "gpqa": 51.0, "ifeval": 86.0, "mmlu": 85.0, "arena_elo": 1340},
    "qwen2.5-7b":         {"swe_bench": 27.0, "humaneval": 72.0, "gpqa": 35.0, "ifeval": 74.0, "mmlu": 74.2, "arena_elo": 1200},
    "qwen2.5-14b":        {"swe_bench": 33.0, "humaneval": 78.0, "gpqa": 41.0, "ifeval": 79.0, "mmlu": 79.0, "arena_elo": 1230},
    "qwen2.5-32b":        {"swe_bench": 40.0, "humaneval": 84.0, "gpqa": 47.0, "ifeval": 83.0, "mmlu": 83.5, "arena_elo": 1275},
    "qwen2.5-72b":        {"swe_bench": 44.0, "humaneval": 86.0, "gpqa": 52.0, "ifeval": 86.5, "mmlu": 86.5, "arena_elo": 1310},
    "deepseek-r1-distill-qwen-7b":  {"swe_bench": 39.2, "humaneval": 82.5, "gpqa": 49.1, "ifeval": 78.4, "mmlu": 79.8, "arena_elo": 1260},
    "deepseek-r1-distill-qwen-14b": {"swe_bench": 43.0, "humaneval": 85.0, "gpqa": 55.0, "ifeval": 82.0, "mmlu": 83.0, "arena_elo": 1290},
    "deepseek-r1-distill-qwen-32b": {"swe_bench": 49.0, "humaneval": 88.0, "gpqa": 61.0, "ifeval": 85.0, "mmlu": 86.0, "arena_elo": 1325},
    "deepseek-r1-distill-llama-8b": {"swe_bench": 38.0, "humaneval": 80.0, "gpqa": 47.0, "ifeval": 77.0, "mmlu": 78.0, "arena_elo": 1250},
    "deepseek-coder":     {"swe_bench": 45.0, "humaneval": 85.0, "gpqa": 40.0, "ifeval": 78.0, "mmlu": 76.0, "arena_elo": 1240},
    "llama-3.3-70b":      {"swe_bench": 49.8, "humaneval": 88.6, "gpqa": 59.2, "ifeval": 87.5, "mmlu": 88.6, "arena_elo": 1320},
    "llama-3.1-8b":       {"swe_bench": 28.0, "humaneval": 72.6, "gpqa": 32.8, "ifeval": 77.0, "mmlu": 73.0, "arena_elo": 1195},
    "llama-3.2-11b":      {"swe_bench": 25.0, "humaneval": 72.0, "gpqa": 34.5, "ifeval": 74.0, "mmlu": 69.0, "arena_elo": 1190},
    "llama-3.2-3b":       {"swe_bench": 15.0, "humaneval": 58.0, "gpqa": 22.0, "ifeval": 64.0, "mmlu": 63.4, "arena_elo": 1120},
    "llama-3.2-1b":       {"swe_bench": 6.0,  "humaneval": 34.0, "gpqa": 14.0, "ifeval": 53.0, "mmlu": 49.3, "arena_elo": 1040},
    "phi-4-mini":         {"swe_bench": 22.0, "humaneval": 74.4, "gpqa": 30.4, "ifeval": 73.0, "mmlu": 67.3, "arena_elo": 1170},
    "phi-4":              {"swe_bench": 34.0, "humaneval": 80.2, "gpqa": 48.5, "ifeval": 82.0, "mmlu": 81.3, "arena_elo": 1250},
    "phi-3.5-mini":       {"swe_bench": 24.0, "humaneval": 68.0, "gpqa": 34.0, "ifeval": 71.0, "mmlu": 69.0, "arena_elo": 1175},
    "mistral-7b":         {"swe_bench": 20.0, "humaneval": 64.0, "gpqa": 30.0, "ifeval": 68.0, "mmlu": 64.0, "arena_elo": 1160},
    "mistral-nemo":       {"swe_bench": 25.0, "humaneval": 70.0, "gpqa": 34.0, "ifeval": 72.0, "mmlu": 68.0, "arena_elo": 1185},
    "gemma-2-9b":         {"swe_bench": 26.0, "humaneval": 71.0, "gpqa": 38.0, "ifeval": 73.0, "mmlu": 71.3, "arena_elo": 1210},
    "gemma-2-27b":        {"swe_bench": 34.0, "humaneval": 79.0, "gpqa": 45.0, "ifeval": 80.0, "mmlu": 75.2, "arena_elo": 1248},
    "starcoder2":         {"swe_bench": 30.0, "humaneval": 70.0, "gpqa": 26.0, "ifeval": 62.0, "mmlu": 58.0, "arena_elo": 1150},
}


# ─── Live LMArena ratings ──────────────────────────────────────────────────────
# lmarena-ai/leaderboard-dataset is refreshed by LMArena several times a week. We read the
# "overall" rows of the text, webdev (coding) and vision leaderboards through the HF
# datasets-server JSON API, keep open-weight models only, and cache them for a day.
# A snapshot bundled with the app is used on first run / offline.

ARENA_CACHE = CACHE_DIR / "arena.json"
ARENA_SNAPSHOT = Path(__file__).resolve().parent / "data" / "arena_snapshot.json"
ARENA_TTL = 86400
_ROWS_API = "https://datasets-server.huggingface.co/rows"
_BOARDS = ("text", "webdev", "vision")

_arena: Optional[Dict[str, Any]] = None
_arena_lock = threading.Lock()
_refreshing = False


def fetch_arena(max_pages: int = 25) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ts": time.time(), "published": ""}
    with httpx.Client(timeout=30, follow_redirects=True) as c:
        for board in _BOARDS:
            ratings: Dict[str, float] = {}
            for page in range(max_pages):
                for attempt in range(4):
                    r = c.get(_ROWS_API, params={"dataset": "lmarena-ai/leaderboard-dataset", "config": board,
                                                 "split": "latest", "offset": page * 100, "length": 100})
                    if r.status_code == 200:
                        break
                    time.sleep(1.5 * (attempt + 1))
                r.raise_for_status()
                rows = [x["row"] for x in r.json().get("rows", [])]
                overall = [x for x in rows if x.get("category") == "overall"]
                for x in overall:
                    if (x.get("license") or "Proprietary") != "Proprietary" and x.get("rating"):
                        ratings[x["model_name"].lower()] = round(float(x["rating"]), 1)
                        out["published"] = max(out["published"], x.get("leaderboard_publish_date") or "")
                if not overall or len(rows) < 100:
                    break
            out[board] = ratings
    if not out.get("text"):
        raise RuntimeError("LMArena returned no ratings")
    return out


def _load_file(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _refresh_in_background():
    global _arena, _refreshing

    def run():
        global _arena, _refreshing
        try:
            data = fetch_arena()
            ARENA_CACHE.write_text(json.dumps(data), encoding="utf-8")
            with _arena_lock:
                _arena = data
            logger.info(f"LMArena ratings updated ({len(data['text'])} open models, {data['published']})")
        except Exception as e:
            logger.warning(f"LMArena refresh failed: {e}")
        finally:
            _refreshing = False

    if not _refreshing:
        _refreshing = True
        threading.Thread(target=run, daemon=True).start()


def get_arena(refresh: bool = True) -> Dict[str, Any]:
    """Current ratings without ever blocking on the network: cache → bundled snapshot,
    with a background refresh when the data is older than a day."""
    global _arena
    with _arena_lock:
        if _arena is None:
            _arena = _load_file(ARENA_CACHE) or _load_file(ARENA_SNAPSHOT) or {"ts": 0, "published": ""}
        data = _arena
    if refresh and time.time() - data.get("ts", 0) > ARENA_TTL:
        _refresh_in_background()
    return data


_STRIP = re.compile(r"-(instruct|it|chat|hf|bf16|fp8|nvfp4|fp16|gguf)$")


def _norm_keys(name: str) -> List[str]:
    n = name.lower().split("/")[-1].replace("_", "-").replace(" ", "-")
    keys = [n]
    while True:
        m = _STRIP.sub("", n)
        if m == n:
            break
        n = m
        keys.append(n)
    if n.startswith("meta-"):
        keys.append(n[5:])
    return keys


def _index(ratings: Dict[str, float]) -> Dict[str, float]:
    idx: Dict[str, float] = {}
    for name, r in ratings.items():
        for k in _norm_keys(name):
            idx.setdefault(k, r)
    return idx


def match_rating(ratings_index: Dict[str, float], *names: str) -> Optional[float]:
    for n in names:
        if not n:
            continue
        for k in _norm_keys(n):
            if k in ratings_index:
                return ratings_index[k]
    return None


# ─── Scores for the whole catalog ─────────────────────────────────────────────

def _eff_params(params_b: float, active_b: float) -> float:
    # A mixture-of-experts model behaves roughly like a dense model of sqrt(total × active)
    if active_b and active_b < params_b:
        return max((params_b * active_b) ** 0.5, 0.05)
    return max(params_b, 0.05)


def _year(updated: str) -> float:
    try:
        y, m = int(updated[:4]), int(updated[5:7])
        return min(max(y + (m - 1) / 12.0, 2023.0), 2027.0)
    except Exception:
        return 2025.0


YEAR_SLOPE = 35.0        # rating points per year of newer release (held fixed: upload dates are noisy)
ESTIMATE_PENALTY = 15.0  # unmeasured models rank just below measured ones of the same predicted quality


def _fit(points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """rating ≈ a + b·ln(effective params) + YEAR_SLOPE·(year − 2025), a and b by least squares."""
    import math
    default = (1150.0, 60.0, YEAR_SLOPE)
    if len(points) < 12:
        return default
    xs = [math.log(p) for p, _, _ in points]
    ys = [r - YEAR_SLOPE * (y - 2025.0) for _, y, r in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-9:
        return default
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    b = min(max(b, 30.0), 120.0)
    return my - b * mx, b, YEAR_SLOPE


def rating_to_score(rating: float) -> int:
    """Arena rating → 0–100 'Score' column (≈1000 → 0, ≈1540 → 100)."""
    return int(max(0, min(100, (rating - 1000) / 5.4)))


_cache_key: Optional[Tuple] = None
_cache_val: Dict[str, Benchmarks] = {}


def catalog_benchmarks(models) -> Dict[str, Benchmarks]:
    """Benchmarks for every model id: measured LMArena ratings where the model is on the
    leaderboard, published scores for well-known older models, otherwise an estimate from a
    size/recency fit calibrated on the measured models."""
    global _cache_key, _cache_val
    arena = get_arena()
    key = (arena.get("ts"), len(models), tuple(m.id for m in models[:5]), tuple(m.id for m in models[-5:]))
    if key == _cache_key:
        return _cache_val

    idx = {b: _index(arena.get(b) or {}) for b in _BOARDS}
    published = (arena.get("published") or "")[:10]
    measured: Dict[str, float] = {}
    points = []
    for m in models:
        r = match_rating(idx["text"], m.base_model, m.name, m.id)
        if r:
            measured[m.id] = r
            points.append((_eff_params(m.params_b, m.active_params_b), _year(m.updated), r))
    a, b, c = _fit(points)

    import math
    out: Dict[str, Benchmarks] = {}
    for m in models:
        pub = _published(m.id) or _published(m.base_model or m.name)
        rating = measured.get(m.id)
        est = rating is None
        if est:
            eff = _eff_params(m.params_b, m.active_params_b)
            rating = a + b * math.log(eff) + c * (_year(m.updated) - 2025.0) - ESTIMATE_PENALTY
            rating = min(max(rating, 950.0), 1450.0)
        coding = match_rating(idx["webdev"], m.base_model, m.name, m.id)
        vision = match_rating(idx["vision"], m.base_model, m.name, m.id)
        if not est:
            source = f"LMArena rating · {published}" if published else "LMArena rating"
        elif pub:
            source = "Published benchmarks; rating estimated"
        else:
            source = "Estimated from model size and release date"
        out[m.id] = Benchmarks(
            swe_bench=pub.get("swe_bench") if pub else None,
            humaneval=pub.get("humaneval") if pub else None,
            gpqa=pub.get("gpqa") if pub else None,
            ifeval=pub.get("ifeval") if pub else None,
            mmlu=pub.get("mmlu") if pub else None,
            rating=round(rating, 1),
            arena_elo=None if est else int(round(rating)),
            arena_coding=int(round(coding)) if coding else None,
            arena_vision=int(round(vision)) if vision else None,
            estimated=est,
            source=source,
        )
    _cache_key, _cache_val = key, out
    return out


def _published(name: str) -> Optional[Dict[str, float]]:
    """Published scores only for that exact model (after dropping -instruct/-it suffixes), so a
    community fine-tune never inherits its base model's results."""
    if not name:
        return None
    for k in _norm_keys(name):
        if k in _DB:
            return {x: v for x, v in _DB[k].items() if x != "arena_elo"}
    return None
