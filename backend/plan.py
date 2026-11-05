"""
Plan mode — the question turned around: not "what fits this computer?" but "what does
this model need?" (after llmfit's `plan`, MIT, github.com/AlexsJones/llmfit).

For one model, quantization and context it reports the memory each way of running it
needs (all on the GPU, split between GPU and CPU, CPU only), the speed to expect, whether
this computer can do it and what it would take, how many conversations fit in memory at
once, and which common machines would run it.
"""
import math
from typing import Any, Dict, List, Optional

from .hardware import HardwareProfile
from .model_repository import ModelEntry, QuantOption, estimate_size_gb, quant_quality
from .profiles import PROFILES, profile_hardware_for
from .scoring import (OS_RESERVE_GB, VRAM_RESERVE_GB, _bandwidths, _evaluate, _layers, efficiency,
                      kv_cache_gb)


def _quant(m: ModelEntry, quant: Optional[str]) -> QuantOption:
    if quant:
        q = next((x for x in m.quants if x.quant.lower() == quant.lower()), None)
        if q:
            return q
        if m.format != "mlx":   # a quant the repo doesn't publish: estimate its size
            return QuantOption(quant=quant.upper(), filename="", size_gb=estimate_size_gb(m.params_b, quant.upper()),
                               quality=quant_quality(quant.upper()))
        raise ValueError(f"{m.name} has no {quant} variant")
    # Default: the common 4-bit-ish sweet spot, else the smallest
    for want in ("Q4_K_M", "UD-Q4_K_XL", "4bit"):
        q = next((x for x in m.quants if x.quant == want), None)
        if q:
            return q
    return min(m.quants, key=lambda x: abs(x.size_gb - m.params_b * 0.6))


def _round(x: float) -> float:
    return round(x, 1)


def make_plan(m: ModelEntry, hw: HardwareProfile, ctx: int, quant: Optional[str] = None,
              target_tps: Optional[float] = None) -> Dict[str, Any]:
    q = _quant(m, quant)
    native = m.context_k * 1024
    ctx = max(512, min(int(ctx), native))
    weights = q.size_gb
    kv = kv_cache_gb(m, ctx)
    buffers = 0.3 + (0.6 if m.is_vision else 0.0)
    total = (weights + kv + buffers) * 1.05
    active = weights * min(1.0, (m.active_params_b or m.params_b) / max(m.params_b, 0.01))
    eff = efficiency()
    gpu_bw, cpu_bw = _bandwidths(hw)
    g = hw.gpu
    unified = g.integrated or g.vendor == "Apple"
    free_vram = max(0.0, g.free_vram_gb - (0 if unified else VRAM_RESERVE_GB))
    free_ram = max(0.0, hw.available_ram_gb - OS_RESERVE_GB)
    gpu_pool = g.free_vram_gb if g.vendor == "Apple" else (free_ram if unified else free_vram)

    def tps(gpu_frac: float) -> float:
        t = active * gpu_frac / (gpu_bw * eff) + active * (1 - gpu_frac) / (cpu_bw * eff)
        return round(min(250.0, 1.0 / t), 1) if t > 0 else 0.0

    def grade(need: float, have: float) -> str:
        if need > have:
            return "No Fit"
        pct = need / max(have, 0.1)
        return "Perfect" if pct <= 0.8 else "Good" if pct <= 0.95 else "Marginal"

    def cores_for(rate: float) -> int:
        # Decode is memory-bound; cores matter mostly for prompt processing and CPU paths
        return max(4, min(32, math.ceil(rate / 4) * 2)) if rate else 4

    paths: List[Dict[str, Any]] = []
    gpu_tps = tps(1.0)
    paths.append({
        "path": "GPU", "label": "Everything on the graphics card" if not unified else "Unified memory (GPU)",
        "min_vram_gb": _round(total), "rec_vram_gb": _round(total * 1.2),
        "min_ram_gb": _round(max(8.0, total * 0.2)), "rec_ram_gb": _round(max(12.0, total * 0.25)),
        "cores": cores_for(0), "tok_s": gpu_tps, "fit_here": grade(total, gpu_pool),
        "gpu_layers": _layers(m.params_b),
    })
    if not unified and g.total_vram_gb > 0:
        on_gpu = max(0.0, min(1.0, (free_vram - kv - buffers) / max(weights, 0.1)))
        split_need_ram = max(0.0, total - free_vram) + 1.0
        paths.append({
            "path": "CPU+GPU", "label": "Split between graphics card and system memory",
            "min_vram_gb": _round(min(total, kv + buffers + 2.0)), "rec_vram_gb": _round(min(total, kv + buffers + 6.0)),
            "min_ram_gb": _round(total), "rec_ram_gb": _round(total * 1.2), "cores": cores_for(8),
            "tok_s": tps(on_gpu), "fit_here": grade(split_need_ram, free_ram) if total <= free_vram + free_ram else "No Fit",
            "gpu_layers": int(_layers(m.params_b) * on_gpu), "gpu_share_pct": int(on_gpu * 100),
        })
    paths.append({
        "path": "CPU", "label": "Processor and system memory only",
        "min_vram_gb": 0.0, "rec_vram_gb": 0.0, "min_ram_gb": _round(total), "rec_ram_gb": _round(total * 1.2),
        "cores": cores_for(16), "tok_s": tps(0.0),
        "fit_here": "Marginal" if grade(total, free_ram) in ("Perfect", "Good") else grade(total, free_ram),
        "gpu_layers": 0,
    })

    # What this computer is missing for the fastest path
    upgrade = None
    if paths[0]["fit_here"] == "No Fit":
        short = total - gpu_pool
        upgrade = {"memory_gb": _round(short), "pool": "unified memory" if unified else "graphics memory",
                   "message": f"About {short:.1f} GB more {'memory' if unified else 'graphics memory'} "
                              f"would run it fully on the GPU."}

    # Conversations that fit at once: each needs its own KV cache at this context
    best = next((p for p in paths if p["fit_here"] != "No Fit"), None)
    pool = gpu_pool if best and best["path"] == "GPU" else free_ram + (free_vram if not unified else 0)
    budget = pool - (weights + buffers) * 1.05
    sessions = max(0, int(budget // max(kv * 1.05, 1e-3))) if best else 0

    needed_bw = None
    if target_tps and target_tps > 0:
        needed_bw = round(active * target_tps / eff)

    machines = []
    for p in PROFILES:
        phw = profile_hardware_for(p["id"])
        r = _evaluate(m, q, phw, ctx)
        machines.append({"id": p["id"], "label": p["label"], "fit": r.fit, "mode": r.mode,
                         "tok_s": r.tok_s, "mem_pct": r.mem_pct,
                         "meets_target": bool(target_tps) and r.fit != "No Fit" and r.tok_s >= target_tps})
    order = {"Perfect": 0, "Good": 1, "Marginal": 2, "No Fit": 3}
    machines.sort(key=lambda x: (order[x["fit"]], -x["tok_s"]))

    return {
        "model_id": m.id, "name": m.name, "quant": q.quant, "quants": [x.quant for x in m.quants],
        "context": ctx, "native_context": native, "weights_gb": _round(weights), "kv_cache_gb": round(kv, 2),
        "total_gb": _round(total), "paths": paths, "upgrade": upgrade, "parallel_sessions": sessions,
        "target_tps": target_tps, "needed_bandwidth_gbps": needed_bw,
        "this_computer": {"gpu": g.name, "gpu_bandwidth_gbps": round(gpu_bw), "simulated": hw.simulated},
        "machines": machines,
    }
