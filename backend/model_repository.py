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
from concurrent.futures import ThreadPoolExecutor
from .utils import CACHE_DIR, get_logger, hf_headers

logger = get_logger("model_repo")
CACHE_FILE = CACHE_DIR / "models_v4.json"
CUSTOM_FILE = CACHE_DIR / "custom_models.json"
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

FULL_PRECISION = {"BF16", "F16", "F32", "FP16"}

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
    base_model: str = ""          # original repo the GGUF was made from, e.g. Qwen/Qwen3-8B
    publisher: str = ""           # who made the GGUF (unsloth, bartowski, …)
    sources: List[str] = []       # every GGUF repo of this base model, preferred first
    gated: bool = False           # needs a HuggingFace token + accepted license
    custom: bool = False          # added by the user from a HuggingFace link
    format: str = "gguf"          # gguf (llama.cpp) | mlx (Apple MLX, one repo per quantization)


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


_ORG_PREFIX = re.compile(r"^[A-Za-z0-9.-]+_(?=[A-Za-z])")


def _display_stem(rid: str, base_model: str = "") -> str:
    """Human name stem: the repo name without -GGUF and without bartowski's 'Org_' prefix
    ('bartowski/Qwen_Qwen3-8B-GGUF' -> 'Qwen3-8B'). The base model only breaks ties when the
    repo name is not descriptive."""
    stem = re.sub(r"-GGUF$", "", rid.split("/")[-1], flags=re.I)
    stem = _ORG_PREFIX.sub("", stem)
    if base_model and len(stem) < 3:
        return base_model.split("/")[-1]
    return stem


def _entry(rid: str, params_b: float, context: int, files: List[str], is_vision: bool,
           downloads: int = 0, likes: int = 0, updated: str = "", arch: str = "",
           base_model: str = "", gated: bool = False) -> Optional[ModelEntry]:
    rname = _display_stem(rid, base_model)
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
        provider=_derive_provider(base_model or rname),
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
        base_model=base_model,
        publisher=rid.split("/")[0],
        sources=[rid],
        gated=gated,
    )
    m["use_case"] = _derive_use_case(m)
    return ModelEntry(**m)


def _base_model_of(item: Dict) -> str:
    tags = item.get("tags") or []
    for t in tags:
        if t.startswith("base_model:quantized:"):
            return t.split(":", 2)[2]
    for t in tags:
        if t.startswith("base_model:") and t.count(":") == 1:
            return t.split(":", 1)[1]
    return ""


# Repos that load in llama.cpp but aren't chat models (speech, embeddings, rerankers, raw exports)
_NON_CHAT_NAME = re.compile(r"(^|[-_.])(tts|asr|whisper|embed|embedding|embeddings|rerank|reranker|"
                            r"unquantized|transformers|vae|speech|audio)([-_.]|$)", re.I)


def _from_hf_item(item: Dict, require_meta: bool = True) -> Optional[ModelEntry]:
    rid = item.get("id", "")
    if not REPO_ID_RE.match(rid):
        return None
    if require_meta and _NON_CHAT_NAME.search(rid.split("/")[-1]):
        return None
    gguf = item.get("gguf") or {}
    arch = (gguf.get("architecture") or "").lower()
    if item.get("pipeline_tag") in _NON_CHAT_PIPELINES or arch in _NON_CHAT_ARCH:
        return None
    context = gguf.get("context_length")
    if not context and require_meta:
        return None  # diffusion/image models carry no context length
    base = _base_model_of(item)
    total = gguf.get("total")
    params_b = total / 1e9 if total else _params_from_name(_display_stem(rid, base))
    if not params_b:
        return None
    files = [s.get("rfilename", "") for s in item.get("siblings") or []]
    is_vision = any("mmproj" in f.lower() for f in files)
    try:
        return _entry(rid, params_b, int(context or 8192), files, is_vision,
                      downloads=item.get("downloads") or 0, likes=item.get("likes") or 0,
                      updated=item.get("lastModified") or "", arch=arch, base_model=base,
                      gated=bool(item.get("gated")))
    except Exception as ex:
        logger.debug(f"Skipping {rid}: {ex}")
        return None


def _from_hf(data: List[Dict]) -> List[ModelEntry]:
    results: List[ModelEntry] = []
    for item in data:
        rid = item.get("id", "")
        if not rid.upper().endswith("-GGUF") or re.search(r"-MTP-GGUF$", rid, re.I):
            continue  # MTP repos duplicate the base repo with speculative-decoding heads
        e = _from_hf_item(item)
        if e:
            results.append(e)
    return results


def _dedupe_key(m: ModelEntry) -> str:
    if m.base_model:
        return m.base_model.lower()
    return re.sub(r"[\s_-]+", "-", m.name.lower())


def merge_sources(models: List[ModelEntry]) -> List[ModelEntry]:
    """One entry per base model. `models` must be ordered by source priority; the first
    repo seen becomes the entry, the others are recorded as alternative sources."""
    merged: Dict[str, ModelEntry] = {}
    for m in models:
        k = _dedupe_key(m)
        if k not in merged:
            merged[k] = m
        else:
            keep = merged[k]
            if m.id not in keep.sources:
                keep.sources.append(m.id)
            keep.downloads += m.downloads
    return list(merged.values())


# ─── MLX models (mlx-community) ────────────────────────────────────────────────
# MLX publishes one repository per quantization (Qwen3-8B-4bit, Qwen3-8B-8bit, …), each a
# folder of .safetensors + config/tokenizer files. They are grouped into one entry per base
# model; each QuantOption.filename holds the repo id of that variant.

_MLX_SUFFIX = re.compile(
    r"[-_]((?:\d+(?:\.\d+)?bit)(?:[-_][A-Za-z0-9.]+)*|bf16|fp16|fp8|mxfp4(?:[-_][\w]+)*|mxfp8(?:[-_][\w]+)*|"
    r"nvfp4(?:[-_][\w]+)*|mixed[-_][\d_]+[-\w]*|dwq[-\w]*)$", re.I)


def mlx_quant_label(repo_name: str, config: Optional[Dict] = None) -> Tuple[str, str]:
    """('Qwen3-8B-4bit-DWQ') -> ('Qwen3-8B', '4bit-DWQ'); falls back to the config's bit width."""
    m = _MLX_SUFFIX.search(repo_name)
    if m:
        return repo_name[:m.start()], m.group(1)
    bits = ((config or {}).get("quantization_config") or (config or {}).get("quantization") or {}).get("bits")
    stem = re.sub(r"[-_]mlx$", "", repo_name, flags=re.I)
    return stem, (f"{bits}bit" if bits else "bf16")


def mlx_bits(label: str) -> float:
    low = label.lower()
    if low.startswith(("bf16", "fp16")):
        return 16.0
    if low.startswith(("fp8", "mxfp8")):
        return 8.25
    if low.startswith(("mxfp4", "nvfp4")):
        return 4.25
    m = re.match(r"(?:mixed[-_])?(\d+(?:\.\d+)?)", low)
    bits = float(m.group(1)) if m else 4.0
    return bits + 0.5  # per-group scales and biases (group size 64)


def mlx_quality(label: str) -> float:
    b = mlx_bits(label) - 0.5
    q = 100.0 if b >= 15 else 99.5 if b >= 7.9 else 98.0 if b >= 5.9 else 96.5 if b >= 4.9 else \
        94.0 if b >= 3.9 else 88.0 if b >= 2.9 else 78.0
    if re.search(r"dwq|awq", label, re.I):
        q += 1.5  # distilled / activation-aware quants recover part of the loss
    return min(100.0, q)


def _mlx_files_ok(files: List[str]) -> bool:
    low = [f.lower() for f in files]
    return any(f.endswith(".safetensors") for f in low) and "config.json" in low and \
        not any(f.endswith(".gguf") for f in low)


def _fetch_mlx(c: httpx.Client, context_by_base: Dict[str, int]) -> List[ModelEntry]:
    params = [("author", "mlx-community"), ("limit", "300"), ("sort", "downloads"),
              ("pipeline_tag", "text-generation")]
    params += [("expand[]", x) for x in ("siblings", "downloads", "likes", "lastModified", "tags",
                                         "safetensors", "config", "gated")]
    r = c.get(f"{HF_API}/models", params=params)
    r.raise_for_status()
    return mlx_entries(r.json(), context_by_base)


def mlx_entries(items: List[Dict], context_by_base: Dict[str, int]) -> List[ModelEntry]:
    groups: Dict[str, Dict[str, Any]] = {}
    for item in items:
        rid = item.get("id", "")
        name = rid.split("/")[-1]
        if not REPO_ID_RE.match(rid) or _NON_CHAT_NAME.search(name):
            continue
        files = [s.get("rfilename", "") for s in item.get("siblings") or []]
        if not _mlx_files_ok(files):
            continue
        total = (item.get("safetensors") or {}).get("total")
        stem, label = mlx_quant_label(name, item.get("config"))
        params_b = total / 1e9 if total else _params_from_name(stem)
        if not params_b:
            continue
        base = _base_model_of(item)
        key = (base or stem).lower()
        g = groups.setdefault(key, {"stem": stem, "base": base, "variants": [], "downloads": 0,
                                    "likes": 0, "updated": "", "gated": False})
        g["variants"].append((label, rid, params_b))
        g["downloads"] += item.get("downloads") or 0
        g["likes"] = max(g["likes"], item.get("likes") or 0)
        g["updated"] = max(g["updated"], (item.get("lastModified") or "")[:10])
        g["gated"] = g["gated"] or bool(item.get("gated"))

    out: List[ModelEntry] = []
    for g in groups.values():
        seen = set()
        quants = []
        for label, rid, params_b in g["variants"]:
            if label.lower() in seen:
                continue
            seen.add(label.lower())
            quants.append(QuantOption(quant=label, filename=rid,
                                      size_gb=round(params_b * 1e9 * mlx_bits(label) / 8 / 1024 ** 3 + 0.05, 2),
                                      quality=mlx_quality(label)))
        quants.sort(key=lambda q: q.size_gb)
        # Prefer a plain 4-bit repo as the entry's id (most common, good default)
        primary = next((q for q in quants if q.quant.lower() == "4bit"), quants[0])
        params_b = next(p for l, r, p in g["variants"] if r == primary.filename)
        stem = g["stem"]
        active, is_moe = _active_params(stem, params_b)
        ctx_k = context_by_base.get(g["base"].lower(), 32) if g["base"] else 32
        m = dict(
            id=primary.filename, name=stem.replace("-", " ").replace("_", " "),
            provider=_derive_provider(g["base"] or stem), family="mlx",
            params_b=round(params_b, 2), active_params_b=round(active, 2), context_k=ctx_k,
            is_vision=False, is_coding=bool(_CODING_RE.search(stem)),
            is_reasoning=bool(_REASON_RE.search(stem)), is_moe=is_moe, quants=quants,
            downloads=g["downloads"], likes=g["likes"], updated=g["updated"],
            hf_url=f"https://huggingface.co/{primary.filename}", base_model=g["base"],
            publisher="mlx-community", sources=[q.filename for q in quants], gated=g["gated"],
            format="mlx",
        )
        m["use_case"] = _derive_use_case(m)
        out.append(ModelEntry(**m))
    return out


def mlx_variants(entry: ModelEntry) -> List[Dict[str, Any]]:
    """Download options of an MLX entry: one per variant repo, with exact sizes."""
    out = []
    for q in entry.quants:
        files = [f for f in repo_tree(q.filename) if mlx_wanted_file(f["path"])]
        size = sum(f["size"] for f in files)
        out.append({"quant": q.quant, "files": files, "size_bytes": size,
                    "size_gb": round(size / 1024 ** 3, 2), "quality": q.quality, "shards": 1,
                    "repo": q.filename})
    out.sort(key=lambda g: g["size_bytes"])
    return out


def mlx_wanted_file(path: str) -> bool:
    low = path.lower()
    name = low.split("/")[-1]
    if name in (".gitattributes",) or name.startswith("readme") or low.startswith(("original/", ".")):
        return False
    return not name.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".md"))


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

# GGUF publishers, in order of preference when several host the same base model
SOURCES: List[Tuple[str, int, int]] = [
    # (author, how many repos to scan, minimum downloads)
    ("unsloth", 200, 0),
    ("bartowski", 150, 3000),
    ("lmstudio-community", 100, 3000),
    ("ggml-org", 60, 1000),
]
_EXPAND = ("gguf", "siblings", "downloads", "likes", "lastModified", "pipeline_tag", "tags", "gated")


def _fetch_author(c: httpx.Client, author: str, limit: int, min_dl: int) -> List[ModelEntry]:
    params = [("author", author), ("search", "GGUF"), ("limit", str(limit)), ("sort", "downloads")]
    params += [("expand[]", x) for x in _EXPAND]
    r = c.get(f"{HF_API}/models", params=params)
    r.raise_for_status()
    data = [i for i in r.json() if (i.get("downloads") or 0) >= min_dl]
    return _from_hf(data)


def _fetch_hf() -> List[ModelEntry]:
    per_source: List[List[ModelEntry]] = []
    with httpx.Client(timeout=HF_TIMEOUT, follow_redirects=True, headers=hf_headers()) as c:
        with ThreadPoolExecutor(max_workers=len(SOURCES)) as ex:
            futures = [ex.submit(_fetch_author, c, a, n, d) for a, n, d in SOURCES]
            for (author, _, _), f in zip(SOURCES, futures):
                try:
                    per_source.append(f.result())
                except Exception as e:
                    logger.warning(f"HF fetch for {author} failed: {e}")
                    per_source.append([])
    if not any(per_source):
        raise RuntimeError("no catalog data")
    gguf = merge_sources([m for group in per_source for m in group])
    context_by_base = {m.base_model.lower(): m.context_k for m in gguf if m.base_model}
    try:
        with httpx.Client(timeout=HF_TIMEOUT, follow_redirects=True, headers=hf_headers()) as c:
            mlx = _fetch_mlx(c, context_by_base)
    except Exception as e:
        logger.warning(f"MLX catalog fetch failed: {e}")
        mlx = []
    return gguf + mlx


# ─── User-added models (paste a HuggingFace link) ──────────────────────────────

def _read_custom() -> List[ModelEntry]:
    try:
        return [ModelEntry(**m) for m in json.loads(CUSTOM_FILE.read_text(encoding="utf-8"))]
    except Exception:
        return []


def _write_custom(entries: List[ModelEntry]):
    CUSTOM_FILE.write_text(json.dumps([e.model_dump() for e in entries], indent=1), encoding="utf-8")


_URL_RE = re.compile(r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/([^/?#\s]+)/([^/?#\s]+)")


def parse_repo_id(text: str) -> str:
    """Accept 'owner/name', 'hf.co/owner/name' or any huggingface.co URL inside the repo."""
    t = (text or "").strip()
    m = _URL_RE.match(t)
    rid = f"{m.group(1)}/{m.group(2)}" if m else t.strip("/")
    if not REPO_ID_RE.match(rid) or rid.split("/")[0] in ("datasets", "spaces", "models"):
        raise ValueError("Enter a HuggingFace model link like https://huggingface.co/owner/model-GGUF")
    return rid


def add_custom_model(text: str) -> ModelEntry:
    global _mem_cache
    rid = parse_repo_id(text)
    params = [("expand[]", x) for x in _EXPAND + ("safetensors", "config")]
    with httpx.Client(timeout=HF_TIMEOUT, follow_redirects=True, headers=hf_headers()) as c:
        r = c.get(f"{HF_API}/models/{rid}", params=params)
    if r.status_code in (401, 403):
        raise PermissionError("This repository is private or gated. Add a HuggingFace token in Settings.")
    if r.status_code == 404:
        raise FileNotFoundError(f"Model '{rid}' was not found on HuggingFace")
    r.raise_for_status()
    item = r.json()
    item["id"] = item.get("id") or rid
    e = _from_hf_item(item, require_meta=False)
    if e is None and _mlx_files_ok([s.get("rfilename", "") for s in item.get("siblings") or []]):
        found = mlx_entries([item], {})
        e = found[0] if found else None
    if e is None:
        raise LookupError("No model files found: the repository needs .gguf files (llama.cpp) or "
                          "MLX weights (.safetensors + config.json)")
    e.custom = True
    with _lock:
        _write_custom([x for x in _read_custom() if x.id != e.id] + [e])
    return e


def remove_custom_model(model_id: str) -> bool:
    with _lock:
        custom = _read_custom()
        keep = [x for x in custom if x.id != model_id]
        if len(keep) == len(custom):
            return False
        _write_custom(keep)
        return True


def _with_custom(models: List[ModelEntry]) -> List[ModelEntry]:
    custom = _read_custom()
    ids = {c.id for c in custom}
    return [m for m in models if m.id not in ids] + custom


def get_models(force: bool = False) -> List[ModelEntry]:
    global _mem_cache
    with _lock:
        if not force and _mem_cache and time.time() - _mem_cache[0] < CACHE_TTL:
            return _with_custom(_mem_cache[1])

        if not force and CACHE_FILE.exists():
            try:
                raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                if time.time() - raw.get("ts", 0) < CACHE_TTL and raw.get("models"):
                    entries = [ModelEntry(**m) for m in raw["models"]]
                    _mem_cache = (raw["ts"], entries)
                    return _with_custom(entries)
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
            except Exception:
                logger.info("Using offline fallback model dataset")
                models = _fallback()
            _mem_cache = (time.time() - CACHE_TTL + 600, models)  # retry in 10 min
            return _with_custom(models)

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
        return _with_custom(models)


def find_model(model_id: str) -> Optional[ModelEntry]:
    for m in get_models():
        if m.id == model_id or model_id in m.sources:
            return m
    return None


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
    with httpx.Client(timeout=HF_TIMEOUT, follow_redirects=True, headers=hf_headers()) as c:
        r = c.get(f"{HF_API}/models/{model_id}/tree/main", params={"recursive": "true"})
        if r.status_code == 401 or r.status_code == 403:
            raise PermissionError("This repository is gated or private. Accept its license on "
                                  "HuggingFace and add your access token in Settings.")
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
