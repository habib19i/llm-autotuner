"""
Stable, versioned REST API for scripts, agents and dashboards (shaped after llmfit's
`/api/v1`, MIT, github.com/AlexsJones/llmfit). Same localhost-only guards as the rest of
the app. Every endpoint accepts hardware overrides — `profile`, or `ram_gb` / `vram_gb` /
`cpu_cores` — to answer for another machine.

    GET /api/v1/system
    GET /api/v1/models          filters: min_fit, use_case, provider, search, format,
                                installed, sort, limit
    GET /api/v1/models/top      the best runnable models (min_fit defaults to good)
    GET /api/v1/models/{query}  models whose id or name contains the query
    GET /api/v1/plan            model, context, quant, target_tps
    GET /api/v1/storage         keep, selection, os_reserve, scratch, headroom, perfect, search
    GET /api/v1/benchmarks      measured runs on this machine
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from . import hardware as _hardware
from .hardware import HardwareProfile
from .utils import APP_VERSION

router = APIRouter(prefix="/api/v1", tags=["v1"])

FIT_RANK = {"Perfect": 0, "Good": 1, "Marginal": 2, "No Fit": 3}
MIN_FIT = {"perfect": 0, "good": 1, "marginal": 2, "any": 3, "too_tight": 3}
SORTS = {"score": lambda r: -r.score, "tps": lambda r: -r.tok_s, "params": lambda r: -r.params_b,
         "mem": lambda r: r.mem_pct, "ctx": lambda r: -r.ctx_k, "fit": lambda r: (FIT_RANK[r.fit], -r.score),
         "name": lambda r: r.name.lower()}
USE_CASES = {"coding": lambda r: r.is_coding, "reasoning": lambda r: r.is_reasoning,
             "vision": lambda r: r.is_vision, "multimodal": lambda r: r.is_vision,
             "chat": lambda r: r.use_case == "Chat", "general": lambda r: not r.is_coding}


def hardware(profile: Optional[str] = None, ram_gb: Optional[float] = None, vram_gb: Optional[float] = None,
             cpu_cores: Optional[int] = None) -> HardwareProfile:
    if profile:
        from .profiles import profile_hardware_for
        try:
            return profile_hardware_for(profile)
        except ValueError as e:
            raise HTTPException(404, str(e))
    hw = _hardware.profile_hardware()
    if ram_gb is None and vram_gb is None and cpu_cores is None:
        return hw
    hw = hw.model_copy(deep=True)
    if ram_gb is not None:
        hw.total_ram_gb, hw.available_ram_gb = ram_gb, round(ram_gb * 0.85, 1)
        if hw.gpu.vendor == "Apple":
            hw.gpu.total_vram_gb, hw.gpu.free_vram_gb = round(ram_gb * 0.75, 1), round(ram_gb * 0.64, 1)
    if vram_gb is not None:
        if hw.gpu.total_vram_gb <= 0 or hw.gpu.vendor == "Generic":
            hw.gpu.name, hw.gpu.vendor, hw.gpu.integrated = "Simulated GPU", "NVIDIA", False
        hw.gpu.total_vram_gb, hw.gpu.free_vram_gb = vram_gb, round(vram_gb * 0.95, 1)
    if cpu_cores is not None:
        hw.physical_cores, hw.logical_cores = cpu_cores, cpu_cores * 2
    hw.simulated = "Custom"
    return hw


def _rows(hw):
    from .selector import build_table
    return build_table(hw)


def _filter(rows, min_fit: str, use_case: Optional[str], provider: Optional[str], search: Optional[str],
            fmt: Optional[str], installed: Optional[bool]):
    lim = MIN_FIT.get((min_fit or "any").lower())
    if lim is None:
        raise HTTPException(400, f"min_fit must be one of {', '.join(MIN_FIT)}")
    out = [r for r in rows if FIT_RANK[r.fit] <= lim]
    if use_case:
        f = USE_CASES.get(use_case.lower())
        if f is None:
            raise HTTPException(400, f"use_case must be one of {', '.join(USE_CASES)}")
        out = [r for r in out if f(r)]
    if provider:
        out = [r for r in out if provider.lower() in r.provider.lower()]
    if search:
        q = search.lower()
        out = [r for r in out if q in r.name.lower() or q in r.id.lower() or q in r.provider.lower()
               or q in r.params_label.lower() or q in r.use_case.lower()]
    if fmt:
        out = [r for r in out if r.format == fmt.lower()]
    if installed is not None:
        from .downloader import installed_models
        have = {x["model_id"] for x in installed_models() if x.get("complete")}
        out = [r for r in out if bool(have & set(r.sources)) == installed]
    return out


@router.get("/system")
def v1_system(profile: Optional[str] = None, ram_gb: Optional[float] = None, vram_gb: Optional[float] = None,
              cpu_cores: Optional[int] = None):
    from .scoring import _bandwidths, efficiency, context_cap
    hw = hardware(profile, ram_gb, vram_gb, cpu_cores)
    gpu_bw, cpu_bw = _bandwidths(hw)
    return {"version": APP_VERSION, "hardware": hw, "gpu_bandwidth_gbps": round(gpu_bw),
            "cpu_bandwidth_gbps": round(cpu_bw), "speed_efficiency": efficiency(),
            "context_cap": context_cap() or None}


@router.get("/models")
def v1_models(min_fit: str = "any", use_case: Optional[str] = None, provider: Optional[str] = None,
              search: Optional[str] = None, format: Optional[str] = None, installed: Optional[bool] = None,
              sort: str = "fit", limit: int = Query(0, ge=0, le=5000), profile: Optional[str] = None,
              ram_gb: Optional[float] = None, vram_gb: Optional[float] = None, cpu_cores: Optional[int] = None):
    if sort not in SORTS:
        raise HTTPException(400, f"sort must be one of {', '.join(SORTS)}")
    hw = hardware(profile, ram_gb, vram_gb, cpu_cores)
    rows = sorted(_filter(_rows(hw), min_fit, use_case, provider, search, format, installed), key=SORTS[sort])
    return {"count": len(rows), "simulated": hw.simulated, "models": rows[:limit] if limit else rows}


@router.get("/models/top")
def v1_top(limit: int = Query(5, ge=1, le=100), min_fit: str = "good", use_case: Optional[str] = None,
           provider: Optional[str] = None, format: Optional[str] = None, profile: Optional[str] = None,
           ram_gb: Optional[float] = None, vram_gb: Optional[float] = None, cpu_cores: Optional[int] = None):
    hw = hardware(profile, ram_gb, vram_gb, cpu_cores)
    rows = [r for r in _filter(_rows(hw), min_fit, use_case, provider, None, format, None) if r.fit != "No Fit"]
    rows.sort(key=lambda r: (FIT_RANK[r.fit], -(r.score + min(r.tok_s, 30) * 0.4)))
    return {"count": min(limit, len(rows)), "simulated": hw.simulated, "models": rows[:limit]}


@router.get("/models/{query:path}")
def v1_model(query: str, profile: Optional[str] = None, ram_gb: Optional[float] = None,
             vram_gb: Optional[float] = None, cpu_cores: Optional[int] = None):
    rows = _rows(hardware(profile, ram_gb, vram_gb, cpu_cores))
    q = query.lower()
    exact = [r for r in rows if r.id.lower() == q or q in (s.lower() for s in r.sources)]
    hits = exact or [r for r in rows if q in r.id.lower() or q in r.name.lower()]
    if not hits:
        raise HTTPException(404, f"No model matches '{query}'")
    return {"count": len(hits), "models": hits}


@router.get("/plan")
def v1_plan(model: str, context: int = Query(16384, ge=512, le=1048576), quant: Optional[str] = None,
            target_tps: Optional[float] = Query(None, gt=0), profile: Optional[str] = None,
            ram_gb: Optional[float] = None, vram_gb: Optional[float] = None, cpu_cores: Optional[int] = None):
    from .model_repository import find_model
    from .plan import make_plan
    m = find_model(model)
    if m is None:
        raise HTTPException(404, f"Model not found: {model}")
    try:
        return make_plan(m, hardware(profile, ram_gb, vram_gb, cpu_cores), context, quant, target_tps)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _size(text: str, name: str) -> float:
    t = text.strip().upper().replace(" ", "")
    try:
        if t.endswith("TIB"):
            return float(t[:-3]) * 1099.511627776
        if t.endswith("GIB"):
            return float(t[:-3]) * 1.073741824
        for suf, mul in (("TB", 1000), ("T", 1000), ("GB", 1), ("G", 1)):
            if t.endswith(suf):
                return float(t[:-len(suf)]) * mul
        return float(t)
    except ValueError:
        raise HTTPException(400, f"{name}: not a size ('{text}'); use e.g. 100G or 1T")


@router.get("/storage")
def v1_storage(keep: int = Query(3, ge=1, le=100), selection: str = "score", os_reserve: str = "100G",
               scratch: str = "auto", headroom: float = Query(15, ge=0, lt=100), perfect: bool = False,
               search: str = "", profile: Optional[str] = None, ram_gb: Optional[float] = None,
               vram_gb: Optional[float] = None, cpu_cores: Optional[int] = None):
    from .storage_plan import storage_plan
    hw = hardware(profile, ram_gb, vram_gb, cpu_cores)
    try:
        st = storage_plan(_rows(hw), keep, selection, _size(os_reserve, "os_reserve"),
                          None if scratch.lower() == "auto" else _size(scratch, "scratch"), headroom, perfect, search)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"system": hw, "storage": st}


@router.get("/benchmarks")
def v1_benchmarks():
    from . import bench
    hw = _hardware.profile_hardware()
    return {"hardware": bench.fingerprint(hw), "calibration": bench.calibration(hw),
            "results": bench.for_hardware(hw)}
