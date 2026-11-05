"""
Models already installed in other local runtimes: Ollama and LM Studio (as llmfit shows
them in its "Inst" column). Read-only: their own servers are asked what they have.

    OLLAMA_HOST     default http://127.0.0.1:11434   (GET /api/tags)
    LMSTUDIO_HOST   default http://127.0.0.1:1234    (GET /v1/models; LMSTUDIO_API_KEY if required)
"""
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .utils import get_logger

logger = get_logger("runtimes")
_cache: Tuple[float, Dict[str, Any]] = (0.0, {})
_TTL = 20.0

_DROP = re.compile(r"\b(instruct|it|chat|gguf|mlx|hf|latest|text|base)\b")
_QUANT = re.compile(r"[-_.:](?:ud-)?(?:i?q\d[\w]*|bf16|f16|f32|fp16|\d+bit[\w-]*|mxfp4[\w-]*)$", re.I)


def match_key(name: str) -> str:
    """'qwen3:8b' / 'Qwen3 8B Instruct' / 'lmstudio-community/Qwen3-8B-GGUF' -> 'qwen38b'."""
    n = name.split("/")[-1].lower()
    n = _QUANT.sub("", n)
    n = re.sub(r"[-_:]", " ", n)
    n = _DROP.sub(" ", n)
    return re.sub(r"[^a-z0-9.]", "", n).replace(".", "")


def _host(var: str, default: str) -> str:
    h = os.environ.get(var, "").strip() or default
    return h if h.startswith("http") else "http://" + h


def _get(url: str, headers: Optional[Dict[str, str]] = None) -> Optional[Any]:
    try:
        r = httpx.get(url, timeout=0.8, headers=headers or {})
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def detect(force: bool = False) -> Dict[str, Any]:
    global _cache
    if not force and time.time() - _cache[0] < _TTL:
        return _cache[1]
    out: Dict[str, Any] = {}
    oh = _host("OLLAMA_HOST", "http://127.0.0.1:11434")
    data = _get(f"{oh}/api/tags")
    out["ollama"] = {"name": "Ollama", "url": oh, "running": data is not None,
                     "models": [m.get("name", "") for m in (data or {}).get("models", []) if m.get("name")]}
    lh = _host("LMSTUDIO_HOST", "http://127.0.0.1:1234")
    key = os.environ.get("LMSTUDIO_API_KEY", "")
    data = _get(f"{lh}/v1/models", {"Authorization": f"Bearer {key}"} if key else None)
    out["lmstudio"] = {"name": "LM Studio", "url": lh, "running": data is not None,
                       "models": [m.get("id", "") for m in (data or {}).get("data", []) if m.get("id")]}
    _cache = (time.time(), out)
    return out


def matches(catalog: List[Any], found: Dict[str, Any]) -> Dict[str, List[str]]:
    """{catalog model id: ['Ollama: qwen3:8b', …]} for models another runtime already has."""
    index: Dict[str, List[str]] = {}
    for rt in found.values():
        for name in rt["models"]:
            if name.endswith(":latest"):   # no size in the tag: too ambiguous to match
                continue
            index.setdefault(match_key(name), []).append(f"{rt['name']}: {name}")
    out: Dict[str, List[str]] = {}
    for m in catalog:
        keys = {match_key(m.name)}
        if getattr(m, "base_model", ""):
            keys.add(match_key(m.base_model))
        hits = sorted({h for k in keys if k for h in index.get(k, [])})
        if hits:
            out[m.id] = hits
    return out
