"""
Scoring engine:
  - Estimates memory requirements per model/quant (weights + KV cache + buffers)
  - Determines GPU/CPU execution mode and how many layers to offload
  - Computes Fit status (Perfect / Good / Marginal / No Fit)
  - Estimates tokens/sec from memory bandwidth (decode is bandwidth-bound)
  - Generates per-persona utility score
"""
from typing import Optional
from pydantic import BaseModel
from .hardware import HardwareProfile
from .model_repository import ModelEntry, QuantOption

# Context sizes tried when launching, largest first; the first that fits is used.
CTX_CANDIDATES = (16384, 8192, 4096)
OS_RESERVE_GB = 1.5      # RAM left for the OS / browser when running on CPU
VRAM_RESERVE_GB = 0.4    # driver / display overhead on dedicated GPUs


class MemResult(BaseModel):
    mode: str           # GPU | CPU+GPU | CPU | No Fit
    fit: str            # Perfect | Good | Marginal | No Fit
    mem_pct: int        # % of relevant memory pool used
    tok_s: float        # estimated tokens/second
    gpu_layers: int     # layers offloaded to GPU
    total_req_gb: float
    vram_headroom_gb: float
    ctx: int = 4096     # context size (tokens) to launch with


TOTAL_LAYERS = {
    # rough layer counts per parameter scale
    1: 16, 2: 28, 4: 36, 8: 36, 14: 40, 24: 40, 32: 64, 72: 80, 128: 88, 400: 94, 2000: 61,
}


def _layers(params_b: float) -> int:
    for k in sorted(TOTAL_LAYERS.keys()):
        if params_b <= k:
            return TOTAL_LAYERS[k]
    return 61


def kv_cache_gb(model: ModelEntry, ctx_tokens: int) -> float:
    # fp16 KV with grouped-query attention: ~16 KB per token per billion params,
    # flattening out for very large models (MLA / few KV heads).
    mb_per_token = min(0.32, 0.016 * max(model.params_b, 0.5) ** 0.85)
    return ctx_tokens * mb_per_token / 1024


def _bandwidths(hw: HardwareProfile):
    """(gpu_gbps, cpu_gbps) — rough effective memory bandwidth."""
    cpu = 45.0 if hw.physical_cores >= 6 else 30.0
    g = hw.gpu
    if g.vendor == "Apple":
        gpu = 120.0 if g.total_vram_gb < 24 else 250.0 if g.total_vram_gb < 64 else 400.0
    elif g.integrated:
        gpu = cpu * 1.3  # shares the same DDR bus as the CPU
    elif g.total_vram_gb >= 20:
        gpu = 800.0
    elif g.total_vram_gb >= 12:
        gpu = 450.0
    elif g.total_vram_gb >= 8:
        gpu = 300.0
    else:
        gpu = 180.0
    return gpu, cpu


def _tok_s(active_gb: float, gpu_frac: float, hw: HardwareProfile) -> float:
    """Decode speed: every token streams the active weights through memory once."""
    gpu_bw, cpu_bw = _bandwidths(hw)
    eff = 0.6  # achievable fraction of peak bandwidth
    t = active_gb * gpu_frac / (gpu_bw * eff) + active_gb * (1 - gpu_frac) / (cpu_bw * eff)
    return round(min(250.0, 1.0 / t), 1) if t > 0 else 0.0


def _evaluate(model: ModelEntry, quant: QuantOption, hw: HardwareProfile, ctx: int) -> MemResult:
    model_gb = quant.size_gb
    kv_gb = kv_cache_gb(model, ctx)
    buffers_gb = 0.3 + (0.6 if model.is_vision else 0.0)  # compute buffers (+ mmproj)
    total_gb = round((model_gb + kv_gb + buffers_gb) * 1.05, 2)

    active = model.active_params_b or model.params_b
    active_gb = model_gb * min(1.0, active / max(model.params_b, 0.01))
    total_layers = _layers(model.params_b)

    free_ram = max(0.0, hw.available_ram_gb - OS_RESERVE_GB)
    g = hw.gpu
    no_gpu = g.vendor == "Generic" or g.total_vram_gb <= 0

    if g.integrated or no_gpu:
        # Unified / shared memory: one pool. Apple's Metal budget is already the GPU pool.
        pool = g.free_vram_gb if g.vendor == "Apple" else free_ram
        if total_gb > pool:
            return MemResult(mode="No Fit", fit="No Fit", mem_pct=int(total_gb / max(pool, 0.1) * 100),
                             tok_s=0.0, gpu_layers=0, total_req_gb=total_gb, vram_headroom_gb=0.0, ctx=ctx)
        pct = int(total_gb / max(pool, 0.1) * 100)
        if g.vendor == "Apple":
            mode, gpu_layers, gpu_frac = "GPU", total_layers, 1.0
        elif no_gpu or not g.vulkan:
            mode, gpu_layers, gpu_frac = "CPU", 0, 0.0
        else:
            # iGPU: offload what fits in the GPU's dedicated carve-out; the rest stays on CPU
            gpu_frac = min(1.0, max(0.0, g.total_vram_gb - 0.5) / max(model_gb + kv_gb, 0.1))
            gpu_layers = int(total_layers * gpu_frac)
            mode = "GPU" if gpu_layers >= total_layers else "CPU+GPU" if gpu_layers > 0 else "CPU"
        fit = "Perfect" if pct <= 55 else "Good" if pct <= 80 else "Marginal"
        return MemResult(mode=mode, fit=fit, mem_pct=pct, tok_s=_tok_s(active_gb, gpu_frac, hw),
                         gpu_layers=gpu_layers, total_req_gb=total_gb,
                         vram_headroom_gb=round(pool - total_gb, 2), ctx=ctx)

    free_vram = max(0.0, g.free_vram_gb - VRAM_RESERVE_GB)
    if total_gb <= free_vram:
        pct = int(total_gb / max(free_vram, 0.1) * 100)
        return MemResult(mode="GPU", fit="Perfect" if pct <= 85 else "Good", mem_pct=pct,
                         tok_s=_tok_s(active_gb, 1.0, hw), gpu_layers=total_layers,
                         total_req_gb=total_gb, vram_headroom_gb=round(free_vram - total_gb, 2), ctx=ctx)

    if total_gb <= free_vram + free_ram:
        # Split offload: KV + buffers stay on GPU, remaining VRAM holds a share of the layers
        gpu_frac = max(0.0, min(1.0, (free_vram - kv_gb - buffers_gb) / max(model_gb, 0.1)))
        gpu_layers = int(total_layers * gpu_frac)
        pct = int(total_gb / max(free_vram + free_ram, 0.1) * 100)
        mode = "CPU+GPU" if gpu_layers > 0 else "CPU"
        fit = "Good" if gpu_frac >= 0.75 else "Marginal"
        return MemResult(mode=mode, fit=fit, mem_pct=pct, tok_s=_tok_s(active_gb, gpu_frac, hw),
                         gpu_layers=gpu_layers, total_req_gb=total_gb, vram_headroom_gb=0.0, ctx=ctx)

    return MemResult(mode="No Fit", fit="No Fit",
                     mem_pct=int(total_gb / max(free_vram + free_ram, 0.1) * 100),
                     tok_s=0.0, gpu_layers=0, total_req_gb=total_gb, vram_headroom_gb=0.0, ctx=ctx)


_FIT_RANK = {"Perfect": 0, "Good": 1, "Marginal": 2, "No Fit": 3}


def estimate_memory(model: ModelEntry, quant: QuantOption, hw: HardwareProfile,
                    ctx: Optional[int] = None) -> MemResult:
    """Evaluate at the largest launch context that gives the best fit."""
    native = model.context_k * 1024
    if ctx:
        return _evaluate(model, quant, hw, min(ctx, native))
    best: Optional[MemResult] = None
    for c in CTX_CANDIDATES:
        r = _evaluate(model, quant, hw, min(c, native))
        if best is None or _FIT_RANK[r.fit] < _FIT_RANK[best.fit]:
            best = r
        if r.fit in ("Perfect", "Good"):
            break
    return best


def persona_score(model: ModelEntry, quant: QuantOption, bench, mem: MemResult, persona: str) -> float:
    """Return 0–100 utility score for given persona."""
    if mem.fit == "No Fit":
        return 0.0

    qual = quant.quality / 100.0
    speed = min(1.0, mem.tok_s / 40.0)
    fit_penalty = {"Perfect": 1.0, "Good": 0.9, "Marginal": 0.65, "No Fit": 0.0}.get(mem.fit, 0.5)

    # General intelligence from the arena rating (measured or estimated), 0–1
    base = max(0.0, min(1.0, (bench.rating - 1000) / 540))

    def pub(*vals_weights):
        vals = [(v, w) for v, w in vals_weights if v is not None]
        if not vals:
            return None
        return sum(v * w for v, w in vals) / sum(w for _, w in vals) / 100.0

    p = persona.lower()
    if p == "coding":
        coding = (max(0.0, min(1.0, (bench.arena_coding - 1000) / 600)) if bench.arena_coding
                  else pub((bench.swe_bench, 0.55), (bench.humaneval, 0.45)))
        intel = (base * 0.5 + coding * 0.5) if coding is not None else base
        intel = min(1.0, intel + (0.06 if model.is_coding else 0.0))
        raw = intel * 0.55 + qual * 0.25 + speed * 0.20
    elif p == "vision":
        vis = max(0.0, min(1.0, (bench.arena_vision - 1000) / 400)) if bench.arena_vision else base
        raw = (vis * 0.3 + (0.5 if model.is_vision else 0.0)) + speed * 0.2
    elif p == "stem":
        sci = pub((bench.gpqa, 0.6), (bench.mmlu, 0.4))
        intel = (base * 0.6 + sci * 0.4) if sci is not None else base
        intel = min(1.0, intel + (0.06 if model.is_reasoning else 0.0))
        raw = intel * 0.65 + qual * 0.20 + speed * 0.15
    elif p == "story":
        ctx_f = min(1.0, mem.ctx / 16384.0)
        raw = ctx_f * 0.30 + base * 0.40 + speed * 0.15 + qual * 0.15
    elif p == "instruction":
        ife = pub((bench.ifeval, 1.0))
        intel = (base * 0.5 + ife * 0.5) if ife is not None else base
        raw = intel * 0.50 + speed * 0.30 + qual * 0.20
    else:  # assistant / general
        raw = speed * 0.35 + base * 0.40 + qual * 0.25

    return round(min(1.0, raw) * fit_penalty * 100.0, 1)
