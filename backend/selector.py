"""
Builds the full enriched model table that the frontend renders.
Each row is a ModelRow — matches llmfit columns.
"""
from typing import List, Optional, Tuple
from pydantic import BaseModel
from .hardware import HardwareProfile, profile_hardware
from .model_repository import ModelEntry, QuantOption, get_models, FULL_PRECISION
from .benchmark_provider import Benchmarks, get_benchmarks
from .scoring import MemResult, estimate_memory, persona_score


class ModelRow(BaseModel):
    """One row in the frontend table."""
    id: str
    name: str
    provider: str
    family: str
    params_b: float
    params_label: str      # "7.6B" / "30B (3B active)"
    active_params_b: float
    score: int             # intelligence / benchmark composite 0–100
    tok_s: float
    quant: str             # best-fit quant for current hardware
    filename: str          # repo path of the best-fit quant (first shard)
    all_quants: List[str]
    mode: str              # GPU | CPU+GPU | CPU | No Fit
    mem_pct: int
    ctx_k: int
    ctx_label: str         # "32k"
    fit: str               # Perfect | Good | Marginal | No Fit
    use_case: str
    is_vision: bool
    is_coding: bool
    is_reasoning: bool
    is_moe: bool
    size_gb: float
    hf_url: str
    # Extended detail
    swe_bench: float
    humaneval: float
    gpqa: float
    ifeval: float
    mmlu: float
    arena_elo: int
    bench_estimated: bool
    total_req_gb: float
    gpu_layers: int
    vram_headroom_gb: float
    launch_ctx: int
    downloads: int


class RecommendationDetail(BaseModel):
    row: ModelRow
    why: str
    persona: str
    persona_score: float
    launch_ctx: int
    launch_threads: int
    launch_gpu_layers: int


def _composite_score(bench: Benchmarks, is_coding: bool, is_reasoning: bool) -> int:
    """Single 0–100 score like llmfit's 'Score' column."""
    weights = [
        (bench.mmlu, 0.30),
        (bench.humaneval, 0.25 if is_coding else 0.10),
        (bench.gpqa, 0.20 if is_reasoning else 0.10),
        (bench.ifeval, 0.15),
        (bench.swe_bench, 0.20 if is_coding else 0.05),
    ]
    total_w = sum(w for _, w in weights)
    score = sum(v * w for v, w in weights) / total_w
    return min(100, int(score))


def _bench(m: ModelEntry) -> Benchmarks:
    return get_benchmarks(m.id, m.params_b, m.is_coding, m.is_reasoning, m.active_params_b)


def _candidate_quants(m: ModelEntry) -> List[QuantOption]:
    """Quantized options, best quality first. Full-precision files only if nothing else exists."""
    qs = [q for q in m.quants if q.quant not in FULL_PRECISION] or list(m.quants)
    return sorted(qs, key=lambda q: (q.quality, -q.size_gb), reverse=True)


# Below ~3 bits per weight (IQ3_XXS, Q2_K, IQ2_*, IQ1_*) output quality drops sharply,
# so a model that only fits at such a quant is never reported as a clean fit.
MIN_CLEAN_QUALITY = 86.0


def _quality_capped(q: QuantOption, mem: MemResult) -> MemResult:
    if q.quality < MIN_CLEAN_QUALITY and mem.fit in ("Perfect", "Good"):
        return mem.model_copy(update={"fit": "Marginal"})
    return mem


def estimate(m: ModelEntry, q: QuantOption, hw: HardwareProfile) -> MemResult:
    return _quality_capped(q, estimate_memory(m, q, hw))


def pick_quant(m: ModelEntry, hw: HardwareProfile) -> Tuple[QuantOption, MemResult]:
    """Highest-quality quant that fits well; otherwise the best-fitting smallest one."""
    cands = _candidate_quants(m)
    results = [(q, estimate(m, q, hw)) for q in cands]
    for want in (("Perfect", "Good"), ("Marginal",)):
        for q, mem in results:
            if mem.fit in want:
                return q, mem
    q = min(cands, key=lambda x: x.size_gb)
    return q, estimate(m, q, hw)


def _params_label(m: ModelEntry) -> str:
    def fmt(v: float) -> str:
        if v >= 100:
            return f"{v:.0f}B"
        if v >= 1:
            return f"{v:.1f}B".replace(".0B", "B")
        return f"{v * 1000:.0f}M"
    if m.is_moe and m.active_params_b and m.active_params_b < m.params_b:
        return f"{fmt(m.params_b)}·A{fmt(m.active_params_b)}"
    return fmt(m.params_b)


def _row(m: ModelEntry, q: QuantOption, mem: MemResult, bench: Benchmarks) -> ModelRow:
    return ModelRow(
        id=m.id, name=m.name, provider=m.provider, family=m.family,
        params_b=m.params_b, params_label=_params_label(m),
        active_params_b=m.active_params_b or m.params_b,
        score=_composite_score(bench, m.is_coding, m.is_reasoning),
        tok_s=mem.tok_s, quant=q.quant, filename=q.filename,
        all_quants=[x.quant for x in m.quants],
        mode=mem.mode, mem_pct=mem.mem_pct,
        ctx_k=m.context_k, ctx_label=f"{m.context_k}k" if m.context_k < 1024 else f"{m.context_k // 1024}M",
        fit=mem.fit, use_case=m.use_case,
        is_vision=m.is_vision, is_coding=m.is_coding, is_reasoning=m.is_reasoning, is_moe=m.is_moe,
        size_gb=q.size_gb, hf_url=m.hf_url,
        swe_bench=bench.swe_bench, humaneval=bench.humaneval, gpqa=bench.gpqa,
        ifeval=bench.ifeval, mmlu=bench.mmlu, arena_elo=bench.arena_elo,
        bench_estimated=bench.estimated,
        total_req_gb=mem.total_req_gb, gpu_layers=mem.gpu_layers,
        vram_headroom_gb=mem.vram_headroom_gb, launch_ctx=mem.ctx,
        downloads=m.downloads,
    )


FIT_ORDER = {"Perfect": 0, "Good": 1, "Marginal": 2, "No Fit": 3}


def build_table(hw: Optional[HardwareProfile] = None) -> List[ModelRow]:
    if hw is None:
        hw = profile_hardware()
    rows: List[ModelRow] = []
    for m in get_models():
        q, mem = pick_quant(m, hw)
        rows.append(_row(m, q, mem, _bench(m)))
    rows.sort(key=lambda r: (FIT_ORDER.get(r.fit, 9), -r.score, -r.tok_s))
    return rows


def get_recommendation(model_id: str, persona: str, hw: Optional[HardwareProfile] = None) -> RecommendationDetail:
    if hw is None:
        hw = profile_hardware()
    m = next((x for x in get_models() if x.id == model_id), None)
    if m is None:
        raise ValueError(f"Model not found: {model_id}")

    bench = _bench(m)

    # Choose best quant for persona + hardware. Below ~4 bits quality drops sharply, so
    # only go there when no 4-bit-or-better quant fits at all.
    best_quant, best_mem, best_score = None, None, -1.0
    cands = _candidate_quants(m)
    for pool in ([q for q in cands if q.quality >= 93.0], cands):
        for q in pool:
            mem = estimate(m, q, hw)
            if mem.fit == "No Fit":
                continue
            s = persona_score(m, q, bench, mem, persona)
            if s > best_score:
                best_quant, best_mem, best_score = q, mem, s
        if best_quant is not None:
            break

    if best_quant is None:
        best_quant, best_mem = pick_quant(m, hw)
        best_score = 0.0

    gpu = hw.gpu
    fit_phrase = {
        "Perfect": (f"fits comfortably in your {gpu.name} ({best_mem.mem_pct}% of available memory)"
                    if best_mem.mode == "GPU" else
                    f"fits comfortably in memory ({best_mem.mem_pct}% of what is available)"),
        "Good": f"fits with ~{best_mem.mem_pct}% memory utilization",
        "Marginal": ("needs split CPU+GPU offloading — expect slower generation"
                     if best_mem.mode == "CPU+GPU" else "uses most of your free memory — close other apps first"),
        "No Fit": "exceeds your available memory at every quantization",
    }.get(best_mem.fit, "runs on your system")

    why_parts = [f"{m.name} ({best_quant.quant}, ~{best_quant.size_gb} GB) {fit_phrase}."]
    est = " (estimated)" if bench.estimated else ""
    if m.is_coding:
        why_parts.append(f"Coding strength: {bench.swe_bench}% SWE-bench, {bench.humaneval}% HumanEval{est}.")
    if m.is_reasoning:
        why_parts.append(f"Reasoning: {bench.gpqa}% GPQA{est}.")
    if m.is_moe and m.active_params_b < m.params_b:
        why_parts.append(f"Mixture-of-experts: only {m.active_params_b}B of {m.params_b}B parameters "
                         f"are active per token, so it runs much faster than its size suggests.")
    if m.is_vision:
        why_parts.append("Accepts images (vision projector is downloaded automatically).")
    if best_mem.fit != "No Fit":
        why_parts.append(f"Expected ~{best_mem.tok_s} tok/s with {best_mem.gpu_layers} GPU layers "
                         f"and a {best_mem.ctx:,}-token context.")

    threads = max(1, hw.physical_cores - (1 if hw.physical_cores > 4 else 0))
    row = _row(m, best_quant, best_mem, bench)
    return RecommendationDetail(
        row=row,
        why=" ".join(why_parts),
        persona=persona,
        persona_score=best_score,
        launch_ctx=best_mem.ctx,
        launch_threads=threads,
        launch_gpu_layers=best_mem.gpu_layers,
    )
