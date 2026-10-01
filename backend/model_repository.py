"""
Model Repository — dynamically syncs with HuggingFace Unsloth GGUF models.

Parameter counts and context lengths come from the GGUF metadata HuggingFace exposes
(`expand[]=gguf`), and the available quantizations come from the repo's real file list,
so nothing is guessed from the repo name unless the metadata is missing.
Falls back to a curated offline dataset when the network is unavailable.
"""
import json
import re
import time
import threading
from typing import Any, Dict, List, Optional, Tuple
import httpx
from pydantic import BaseModel
from .utils import CACHE_DIR, get_logger

logger = get_logger("model_repo")
CACHE_FILE = CACHE_DIR / "models_v3.json"
CACHE_TTL = 86400  # 24 hours
HF_API = "https://huggingface.co/api"
HF_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")

# ─── Quantization tables ───────────────────────────────────────────────────────

# Approximate bits per weight of each llama.cpp quant type (incl. scales)
QUANT_BPW: Dict[str, float] = {
    "IQ1_S": 1.56, "IQ1_M": 1.75, "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.5, "IQ2_M": 2.7,
    "Q2_K": 2.96, "Q2_K_L": 3.1, "IQ3_XXS": 3.06, "IQ3_XS": 3.3, "IQ3_S": 3.44, "IQ3_M": 3.66,
    "Q3_K_S": 3.5, "Q3_K_M": 3.91, "Q3_K_L": 4.27, "IQ4_XS": 4.25, "IQ4_NL": 4.5,
    "Q4_0": 4.55, "Q4_1": 5.0, "Q4_K_S": 4.58, "Q4_K_M": 4.89, "MXFP4": 4.25, "MXFP4_MOE": 4.25,
    "Q5_0": 5.54, "Q5_1": 6.0, "Q5_K_S": 5.54, "Q5_K_M": 5.7, "Q6_K": 6.56, "Q8_0": 8.5,
    "BF16": 16.0, "F16": 16.0, "F32": 32.0,
}

# Relative output quality vs. the full-precision model (0–100)
QUANT_QUALITY: Dict[str, float] = {
    "IQ1_S": 62.0, "IQ1_M": 68.0, "IQ2_XXS": 76.0, "IQ2_XS": 78.0, "IQ2_S": 80.0, "IQ2_M": 82.0,
    "Q2_K": 82.0, "Q2_K_L": 84.0, "IQ3_XXS": 86.0, "IQ3_XS": 87.5, "IQ3_S": 88.5, "IQ3_M": 89.5,
    "Q3_K_S": 88.0, "Q3_K_M": 90.0, "Q3_K_L": 91.0, "IQ4_XS": 93.5, "IQ4_NL": 94.0,
    "Q4_0": 93.0, "Q4_1": 94.0, "Q4_K_S": 94.0, "Q4_K_M": 95.0, "MXFP4": 95.0, "MXFP4_MOE": 95.0,
    "Q5_0": 96.0, "Q5_1": 96.5, "Q5_K_S": 96.5, "Q5_K_M": 97.0, "Q6_K": 98.5, "Q8_0": 99.5,
    "BF16": 100.0, "F16": 100.0, "F32": 100.0,
}

FULL_PRECISION = {"BF16", "F16", "F32"}

_QUANT_RE = re.compile(
    r"[-._]((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?))$", re.IGNORECASE)
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})$")


def quant_from_filename(path: str) -> Optional[str]:
    """'Q4_K_M/Model-Q4_K_M-00001-of-00003.gguf' -> 'Q4_K_M'; 'Model-UD-Q4_K_XL.gguf' -> 'UD-Q4_K_XL'."""
    base = path.split("/")[-1]
    if not base.lower().endswith(".gguf"):
        return None
    stem = _SHARD_RE.sub("", base[:-5])
    m = _QUANT_RE.search(stem)
    if not m:
        return None
    q = m.group(1).upper()
    return q


def _base_quant(q: str) -> str:
    """'UD-Q4_K_XL' -> 'Q4_K_M' (closest standard type) for size/quality tables."""
    b = q[3:] if q.startswith("UD-") else q
    if b.endswith("_XL"):
        b = b[:-3] + "_M"
    return b


def quant_bpw(q: str) -> float:
    bpw = QUANT_BPW.get(_base_quant(q), 4.9)
    return bpw + (0.35 if q.endswith("_XL") else 0.0)


def quant_quality(q: str) -> float:
    base = QUANT_QUALITY.get(_base_quant(q), 92.0)
    if q.startswith("UD-"):
        base += 1.0  # Unsloth dynamic quants keep sensitive layers at higher precision
    return min(100.0, base)


def is_model_file(path: str) -> bool:
    """True for loadable model weights (excludes vision projectors, MTP drafts, imatrix…)."""
    low = path.lower()
    base = low.split("/")[-1]
    if not base.endswith(".gguf"):
        return False
    if "mmproj" in base or base.startswith("mtp-") or low.startswith("mtp/") or "imatrix" in base:
        return False
    return quant_from_filename(path) is not None


def estimate_size_gb(params_b: float, quant: str) -> float:
    return round(params_b * 1e9 * quant_bpw(quant) / 8 / 1024 ** 3 + 0.05, 2)


# ─── Data model ────────────────────────────────────────────────────────────────

class QuantOption(BaseModel):
    quant: str          # e.g. Q4_K_M / UD-Q4_K_XL
    filename: str       # repo path of the (first) file
    size_gb: float      # estimated from params × bits-per-weight
    quality: float      # 0–100


class ModelEntry(BaseModel):
    id: str             # unsloth/Qwen3-8B-GGUF
    name: str           # display name
    provider: str       # Alibaba, Meta, Google …
    family: str         # qwen3, llama, gemma3 … (GGUF architecture)
    params_b: float     # total parameters (billions)
    active_params_b: float = 0.0  # parameters touched per token (MoE); == params_b for dense
    context_k: int      # native context in thousands of tokens
    is_vision: bool
    is_coding: bool
    is_reasoning: bool
    is_moe: bool
    quants: List[QuantOption]
    downloads: int
    likes: int
    updated: str        # YYYY-MM-DD
    hf_url: str
    use_case: str = ""  # Chat / Coding / Vision / Reasoning


# ─── Classification helpers ────────────────────────────────────────────────────

_NON_CHAT_PIPELINES = {
    "feature-extraction", "sentence-similarity", "text-to-image", "image-to-image",
    "image-to-video", "text-to-video", "image-text-to-video", "image-text-to-image",
    "text-to-speech", "text-to-audio", "automatic-speech-recognition", "audio-to-audio",
    "text-classification", "token-classification", "fill-mask",
}
_NON_CHAT_ARCH = {"bert", "nomic-bert", "jina-bert-v2", "gemma-embedding", "flux", "ltxv",
                  "wan", "lumina2", "qwen_image", "diffusion-gemma", "sd3", "t5encoder"}

_PROVIDERS: List[Tuple[str, str]] = [
    # Order matters: "DeepSeek-R1-Distill-Qwen" must resolve to DeepSeek, not Alibaba
    ("deepseek", "DeepSeek"), ("kimi", "Moonshot AI"), ("nemotron", "NVIDIA"),
    ("nvidia", "NVIDIA"), ("medgemma", "Google"), ("gemma", "Google"), ("gpt-oss", "OpenAI"),
    ("glm", "Z.ai"), ("minimax", "MiniMax"), ("devstral", "Mistral AI"),
    ("ministral", "Mistral AI"), ("magistral", "Mistral AI"), ("mistral", "Mistral AI"),
    ("phi", "Microsoft"), ("llama", "Meta"), ("qwen", "Alibaba"), ("qwq", "Alibaba"),
    ("smollm", "Hugging Face"), ("lfm", "Liquid AI"), ("granite", "IBM"), ("step", "StepFun"),
    ("ernie", "Baidu"), ("hunyuan", "Tencent"), ("olmo", "Ai2"), ("cohere", "Cohere"),
    ("command", "Cohere"), ("north", "Cohere"), ("starcoder", "BigCode"), ("falcon", "TII"),
    ("seed", "ByteDance"),
]


def _derive_provider(name: str) -> str:
    p = name.lower()
    for key, prov in _PROVIDERS:
        if key in p:
            return prov
    return "Open Source"


def _derive_use_case(m: Dict) -> str:
    if m["is_coding"]:
        return "Coding"
    if m["is_reasoning"]:
        return "Reasoning"
    if m["is_vision"]:
        return "Vision"
    return "Chat"


_SIZE_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([BM])(?![a-z])", re.IGNORECASE)


def _params_from_name(name: str) -> Optional[float]:
    """Exact token match ('Qwen3-235B-A22B' -> 235.0, '0.8B' -> 0.8, '135M' -> 0.135)."""
    best = None
    for num, unit in _SIZE_RE.findall(name.replace("_", "-")):
        v = float(num) / (1000.0 if unit.upper() == "M" else 1.0)
        best = v if best is None else max(best, v)
    return best


_KNOWN_ACTIVE = {"gpt-oss-20b": 3.6, "gpt-oss-120b": 5.1, "deepseek-r1": 37.0, "deepseek-v3": 37.0,
                 "kimi-k2": 32.0, "glm-4.5-air": 12.0, "glm-4.6": 32.0, "glm-4.7-flash": 3.0}


def _active_params(name: str, total_b: float) -> Tuple[float, bool]:
    low = name.lower()
    m = re.search(r"-a(\d+(?:\.\d+)?)b\b", low)
    if m:
        return float(m.group(1)), True
    m = re.search(r"(\d+)b-(\d+)e\b", low)  # Llama-4-Scout-17B-16E: 17B active
    if m:
        return float(m.group(1)), True
    m = re.search(r"-e(\d+(?:\.\d+)?)b\b", low)  # gemma-3n-E4B: ~4B effective (per-layer embeddings)
    if m:
        return min(float(m.group(1)), total_b), False
    for key, act in _KNOWN_ACTIVE.items():
        if key in low:
            return act, True
    return total_b, False


_CODING_RE = re.compile(r"coder|code|devstral|codestral|starcoder|codegemma", re.I)
_REASON_RE = re.compile(r"thinking|reason|-r1|qwq|gpt-oss|magistral|phi-4-reasoning", re.I)


def _entry(rid: str, params_b: float, context: int, files: List[str], is_vision: bool,
           downloads: int = 0, likes: int = 0, updated: str = "", arch: str = "") -> Optional[ModelEntry]:
    rname = rid.split("/")[-1]
    rname = re.sub(r"-GGUF$", "", rname, flags=re.I)
    active, is_moe = _active_params(rname, params_b)
    if arch.endswith("moe") or arch in ("deepseek2", "deepseek4", "glm4moe", "gpt-oss", "llama4"):
        is_moe = True

    # One entry per quant type; keep the first file (sorted -> shard 00001)
    by_quant: Dict[str, str] = {}
    for f in sorted(files):
        if not is_model_file(f):
            continue
        q = quant_from_filename(f)
        if q and q not in by_quant:
            by_quant[q] = f
    if not by_quant:
        return None
    quants = [QuantOption(quant=q, filename=f, size_gb=estimate_size_gb(params_b, q),
                          quality=quant_quality(q)) for q, f in by_quant.items()]
    quants.sort(key=lambda x: x.size_gb)

    m = dict(
        id=rid,
        name=rname.replace("-", " ").replace("_", " "),
        provider=_derive_provider(rname),
        family=arch or rname.split("-")[0].lower(),
        params_b=round(params_b, 2),
        active_params_b=round(active, 2),
        context_k=max(1, int(round(context / 1024))) if context else 8,
        is_vision=is_vision,
        is_coding=bool(_CODING_RE.search(rname)),
        is_reasoning=bool(_REASON_RE.search(rname)),
        is_moe=is_moe,
        quants=quants,
        downloads=downloads,
        likes=likes,
        updated=(updated or "")[:10],
        hf_url=f"https://huggingface.co/{rid}",
    )
    m["use_case"] = _derive_use_case(m)
    return ModelEntry(**m)


def _from_hf(data: List[Dict]) -> List[ModelEntry]:
    results: List[ModelEntry] = []
    for item in data:
        rid = item.get("id", "")
        if not rid.upper().endswith("-GGUF") or not REPO_ID_RE.match(rid):
            continue
        if re.search(r"-MTP-GGUF$", rid, re.I):
            continue  # duplicate of the base repo with speculative-decoding heads
        gguf = item.get("gguf") or {}
        arch = (gguf.get("architecture") or "").lower()
        pipeline = item.get("pipeline_tag")
        if pipeline in _NON_CHAT_PIPELINES or arch in _NON_CHAT_ARCH:
            continue
        context = gguf.get("context_length")
        if not context:
            continue  # diffusion/image models carry no context length
        total = gguf.get("total")
        params_b = total / 1e9 if total else _params_from_name(rid.split("/")[-1])
        if not params_b:
            continue
        files = [s.get("rfilename", "") for s in item.get("siblings") or []]
        is_vision = any("mmproj" in f.lower() for f in files)
        try:
            e = _entry(rid, params_b, int(context), files, is_vision,
                       downloads=item.get("downloads") or 0, likes=item.get("likes") or 0,
                       updated=item.get("lastModified") or "", arch=arch)
            if e:
                results.append(e)
        except Exception as ex:
            logger.debug(f"Skipping {rid}: {ex}")
    return results


# ─── Offline fallback dataset ──────────────────────────────────────────────────

_STD = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]

# (repo, total params B, context tokens, vision, quants)
_FALLBACK_SPEC = [
    ("unsloth/Qwen3-0.6B-GGUF", 0.6, 40960, False, _STD),
    ("unsloth/Qwen3-1.7B-GGUF", 1.72, 40960, False, _STD),
    ("unsloth/Qwen3-4B-Instruct-2507-GGUF", 4.02, 262144, False, _STD),
    ("unsloth/Qwen3-8B-GGUF", 8.19, 40960, False, _STD),
    ("unsloth/Qwen3-14B-GGUF", 14.77, 40960, False, _STD),
    ("unsloth/Qwen3-32B-GGUF", 32.76, 40960, False, _STD),
    ("unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF", 30.53, 262144, False, _STD),
    ("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", 30.53, 262144, False, _STD),
    ("unsloth/Qwen3-VL-8B-Instruct-GGUF", 8.19, 262144, True, _STD),
    ("unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF", 8.19, 131072, False, _STD),
    ("unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF", 14.77, 131072, False, _STD),
    ("unsloth/gpt-oss-20b-GGUF", 20.91, 131072, False, ["Q4_K_M", "Q8_0", "F16"]),
    ("unsloth/gemma-3-4b-it-GGUF", 3.88, 131072, True, _STD),
    ("unsloth/gemma-3-12b-it-GGUF", 11.77, 131072, True, _STD),
    ("unsloth/gemma-3-27b-it-GGUF", 27.01, 131072, True, _STD),
    ("unsloth/Llama-3.2-1B-Instruct-GGUF", 1.24, 131072, False, _STD),
    ("unsloth/Llama-3.2-3B-Instruct-GGUF", 3.21, 131072, False, _STD),
    ("unsloth/Llama-3.1-8B-Instruct-GGUF", 8.03, 131072, False, _STD),
    ("unsloth/Llama-3.3-70B-Instruct-GGUF", 70.55, 131072, False, ["Q2_K", "Q3_K_M", "Q4_K_M"]),
    ("unsloth/Phi-4-mini-instruct-GGUF", 3.84, 131072, False, _STD),
    ("unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF", 23.57, 131072, True, _STD),
    ("unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF", 23.57, 393216, True, _STD),
]


def _fallback() -> List[ModelEntry]:
    out = []
    for rid, params, ctx, vis, qs in _FALLBACK_SPEC:
        stem = rid.split("/")[-1][:-5]
        files = [f"{stem}-{q}.gguf" for q in qs]
        e = _entry(rid, params, ctx, files, vis)
        if e:
            out.append(e)
    return out


# ─── Public API ────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_mem_cache: Optional[Tuple[float, List[ModelEntry]]] = None


def _fetch_hf() -> List[ModelEntry]:
    params = [("author", "unsloth"), ("search", "GGUF"), ("limit", "200"), ("sort", "downloads")]
    params += [("expand[]", x) for x in ("gguf", "siblings", "downloads", "likes",
                                         "lastModified", "pipeline_tag")]
    with httpx.Client(timeout=HF_TIMEOUT, follow_redirects=True) as c:
        r = c.get(f"{HF_API}/models", params=params)
        r.raise_for_status()
        return _from_hf(r.json())


def get_models(force: bool = False) -> List[ModelEntry]:
    global _mem_cache
    with _lock:
        if not force and _mem_cache and time.time() - _mem_cache[0] < CACHE_TTL:
            return _mem_cache[1]

        if not force and CACHE_FILE.exists():
            try:
                raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                if time.time() - raw.get("ts", 0) < CACHE_TTL and raw.get("models"):
                    entries = [ModelEntry(**m) for m in raw["models"]]
                    _mem_cache = (raw["ts"], entries)
                    return entries
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")

        models: List[ModelEntry] = []
        try:
            models = _fetch_hf()
            logger.info(f"Fetched {len(models)} models from HuggingFace")
        except Exception as e:
            logger.warning(f"HF fetch failed: {e}")

        if not models:
            # Prefer a stale cache over the small offline list
            try:
                raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                models = [ModelEntry(**m) for m in raw["models"]]
                logger.info("Using stale model cache (offline)")
                _mem_cache = (time.time() - CACHE_TTL + 600, models)  # retry in 10 min
                return models
            except Exception:
                logger.info("Using offline fallback model dataset")
                models = _fallback()
                _mem_cache = (time.time() - CACHE_TTL + 600, models)
                return models

        ts = time.time()
        try:
            CACHE_FILE.write_text(
                json.dumps({"ts": ts, "models": [m.model_dump() for m in models]},
                           ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
        _mem_cache = (ts, models)
        return models


def find_model(model_id: str) -> Optional[ModelEntry]:
    return next((m for m in get_models() if m.id == model_id), None)


# ─── Real file listing (for downloads) ─────────────────────────────────────────

_tree_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_TREE_TTL = 600


def repo_tree(model_id: str) -> List[Dict[str, Any]]:
    """All files in a HF repo with exact sizes: [{'path': ..., 'size': ...}]."""
    if not REPO_ID_RE.match(model_id):
        raise ValueError("Invalid model id")
    hit = _tree_cache.get(model_id)
    if hit and time.time() - hit[0] < _TREE_TTL:
        return hit[1]
    with httpx.Client(timeout=HF_TIMEOUT, follow_redirects=True) as c:
        r = c.get(f"{HF_API}/models/{model_id}/tree/main", params={"recursive": "true"})
        if r.status_code == 401 or r.status_code == 403:
            raise PermissionError("This repository is gated or private on HuggingFace")
        if r.status_code == 404:
            raise FileNotFoundError("Repository not found on HuggingFace")
        r.raise_for_status()
        files = [{"path": e["path"], "size": int(e.get("size") or 0)}
                 for e in r.json() if e.get("type") == "file"]
    _tree_cache[model_id] = (time.time(), files)
    return files


def quant_files(model_id: str) -> List[Dict[str, Any]]:
    """Group a repo's weight files by quant: [{quant, files:[{path,size}], size_bytes, quality}]."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for f in repo_tree(model_id):
        if is_model_file(f["path"]):
            groups.setdefault(quant_from_filename(f["path"]), []).append(f)
    out = []
    for q, files in groups.items():
        files.sort(key=lambda f: f["path"])
        size = sum(f["size"] for f in files)
        out.append({
            "quant": q,
            "files": files,
            "size_bytes": size,
            "size_gb": round(size / 1024 ** 3, 2),
            "quality": quant_quality(q),
            "shards": len(files),
        })
    out.sort(key=lambda g: g["size_bytes"])
    return out


def mmproj_file(model_id: str) -> Optional[Dict[str, Any]]:
    """The vision projector to pair with a vision model (prefers F16)."""
    cands = [f for f in repo_tree(model_id)
             if "mmproj" in f["path"].lower().split("/")[-1] and f["path"].lower().endswith(".gguf")]
    if not cands:
        return None
    rank = {"F16": 0, "BF16": 1, "Q8_0": 2, "F32": 3}
    return min(cands, key=lambda f: (rank.get(quant_from_filename(f["path"]) or "", 9), f["size"]))
