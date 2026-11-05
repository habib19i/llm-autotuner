"""
Disk sizing for a model library you switch between (after llmfit's `storage`, MIT,
github.com/AlexsJones/llmfit): pick the models you would keep, then add room for the OS,
one download in progress and free-space headroom, and round up to a drive size.
"""
from typing import Any, Dict, List, Optional

GB_PER_GIB = 1.073741824
TIERS = [256, 512, 1000, 2000, 4000, 8000, 16000]   # decimal GB, as drives are sold


def _tier(gb: float) -> Optional[int]:
    return next((t for t in TIERS if t >= gb), None)


def storage_plan(rows: List[Any], keep: int = 3, selection: str = "score", os_reserve_gb: float = 100.0,
                 scratch_gb: Optional[float] = None, headroom_pct: float = 15.0, perfect_only: bool = False,
                 search: str = "") -> Dict[str, Any]:
    if keep < 1:
        raise ValueError("keep must be at least 1")
    if not 0 <= headroom_pct < 100:
        raise ValueError("headroom must be between 0 and 99")
    q = search.strip().lower()
    cands = [r for r in rows if (r.fit == "Perfect" if perfect_only else r.fit != "No Fit")
             and (not q or q in r.name.lower() or q in r.provider.lower() or q in r.params_label.lower())]
    if selection == "largest":
        cands = sorted(cands, key=lambda r: -r.size_gb)
    elif selection != "score":
        raise ValueError("selection must be 'score' or 'largest'")
    chosen, seen = [], set()
    for r in cands:   # rows arrive ranked by fit and quality
        k = (r.base_model or r.name).lower()
        if k in seen:
            continue
        seen.add(k)
        chosen.append(r)
        if len(chosen) == keep:
            break
    models = [{"id": r.id, "name": r.name, "quant": r.quant, "fit": r.fit, "format": r.format,
               "disk_gb": round(r.size_gb * GB_PER_GIB, 1), "score": r.score} for r in chosen]
    library = round(sum(m["disk_gb"] for m in models), 1)
    scratch = round(max((m["disk_gb"] for m in models), default=0.0) if scratch_gb is None else scratch_gb, 1)
    need = round(os_reserve_gb + library + scratch, 1)
    target = round(need / (1 - headroom_pct / 100), 1)
    warnings = []
    if len(models) < keep:
        warnings.append(f"Only {len(models)} model(s) run on this computer with these filters.")
    if _tier(target) is None:
        warnings.append("More than 16 TB: split the library across drives.")
    return {"models": models, "keep": keep, "selection": selection, "library_gb": library,
            "download_scratch_gb": scratch, "os_reserve_gb": os_reserve_gb, "headroom_percent": headroom_pct,
            "need_gb": need, "target_capacity_gb": target, "minimum_ssd_gb": _tier(need),
            "suggested_ssd_gb": _tier(target), "warnings": warnings}
