"""
Resilient downloader:
- Resolves the real files for (model, quant) from the HuggingFace repo tree, including
  multi-part shards stored in sub-folders and the vision projector (mmproj) when needed
- Each model is stored in its own folder:  models/<Repo-GGUF>/<file>.gguf
- No read timeout (only a connect timeout) so slow/large files never time out mid-stream
- Automatic resume via HTTP Range header on disconnect (up to MAX_RETRIES attempts)
- Exponential back-off between retries
- Tmp file is kept across retries (and app restarts) so progress is never lost
"""
import asyncio
import errno
import json
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import httpx
from pydantic import BaseModel
from .model_repository import REPO_ID_RE, quant_files, mmproj_file, find_model, quant_from_filename
from .utils import MODELS_DIR, get_logger, format_bytes, safe_model_path, rel_model_path, hf_headers

logger = get_logger("downloader")

MAX_RETRIES = 8          # total attempts per file (1 initial + 7 retries)
CHUNK_SIZE = 1024 * 512  # 512 KB chunks
MANIFEST = MODELS_DIR / "manifest.json"
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
DISK_RESERVE = 1024 ** 3  # always leave 1 GB free for the OS


class DiskSpaceError(Exception):
    pass


def disk_info() -> Dict[str, Any]:
    u = shutil.disk_usage(MODELS_DIR)
    used = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file())
    return {"path": str(MODELS_DIR), "free_bytes": u.free, "total_bytes": u.total,
            "free_gb": round(u.free / 1024 ** 3, 1), "models_bytes": used,
            "models_gb": round(used / 1024 ** 3, 2)}


class DownloadJob(BaseModel):
    key: str                  # "<model_id>:<quant>"
    model_id: str
    quant: str
    filename: str             # primary file, relative to models/ (what /api/launch takes)
    status: str               # queued | downloading | done | failed | cancelled
    downloaded_bytes: int = 0
    total_bytes: int = 0
    pct: float = 0.0
    speed_mbs: float = 0.0
    eta_s: int = 0
    file_index: int = 0       # 1-based index of the file being fetched
    file_count: int = 0
    error: str = ""
    attempt: int = 0


_jobs: Dict[str, DownloadJob] = {}
_cancel: Dict[str, bool] = {}
_tasks: Set[asyncio.Task] = set()
_manifest_lock = threading.Lock()


# ─── Manifest (maps local files back to their HF model + quant) ────────────────

def _read_manifest() -> Dict[str, Dict[str, Any]]:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _update_manifest(rel: str, entry: Optional[Dict[str, Any]]):
    with _manifest_lock:
        m = _read_manifest()
        if entry is None:
            m.pop(rel, None)
        else:
            m[rel] = entry
        try:
            MANIFEST.write_text(json.dumps(m, indent=1), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Manifest write failed: {e}")


# ─── Queries ───────────────────────────────────────────────────────────────────

def all_jobs() -> List[DownloadJob]:
    return list(_jobs.values())


def _shard_siblings(p: Path) -> List[Path]:
    """All parts of a split model given its first shard (or just [p])."""
    m = _SHARD_RE.search(p.name)
    if not m:
        return [p]
    prefix = p.name[:m.start()]
    count = int(m.group(2))
    return [p.with_name(f"{prefix}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]


def installed_models() -> List[Dict[str, Any]]:
    manifest = _read_manifest()
    out = []
    for f in sorted(MODELS_DIR.rglob("*.gguf")):
        name = f.name.lower()
        if "mmproj" in name:
            continue
        m = _SHARD_RE.search(f.name)
        if m and int(m.group(1)) != 1:
            continue  # list split models once, by their first shard
        parts = _shard_siblings(f)
        complete = all(p.exists() for p in parts)
        size = sum(p.stat().st_size for p in parts if p.exists())
        rel = rel_model_path(f)
        meta = manifest.get(rel, {})
        if meta.get("total_bytes") and size < meta["total_bytes"]:
            complete = False
        out.append({
            "filename": rel,
            "name": f.name,
            "model_id": meta.get("model_id", ""),
            "quant": meta.get("quant") or quant_from_filename(f.name) or "",
            "size": format_bytes(size),
            "size_bytes": size,
            "shards": len(parts),
            "complete": complete,
            "has_mmproj": any("mmproj" in x.name.lower() for x in f.parent.glob("*.gguf")),
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)),
        })
    return out


def cancel_job(key: str) -> bool:
    if key in _jobs:
        _cancel[key] = True
        return True
    return False


def delete_model(rel: str) -> bool:
    p = safe_model_path(rel)
    if not p.exists() or p.suffix.lower() != ".gguf":
        return False
    for part in _shard_siblings(p):
        for f in (part, part.with_name(part.name + ".tmp")):
            if f.exists():
                f.unlink()
    _update_manifest(rel, None)
    # Remove the model folder once no weights are left in it (drops the mmproj too)
    folder = p.parent
    if folder != MODELS_DIR.resolve():
        remaining = [x for x in folder.glob("*.gguf") if "mmproj" not in x.name.lower()]
        if not remaining:
            for x in folder.iterdir():
                if x.is_file():
                    x.unlink()
            try:
                folder.rmdir()
            except OSError:
                pass
    return True


# ─── Download engine ───────────────────────────────────────────────────────────

# httpx timeout: connect in 15s, but NO read timeout (None) so streams never die
_TIMEOUT = httpx.Timeout(connect=15.0, read=None, write=None, pool=None)


def _local_folder(model_id: str) -> Path:
    return safe_model_path(model_id.split("/")[-1])


def _hf_url(model_id: str, path: str) -> str:
    return f"https://huggingface.co/{model_id}/resolve/main/{path}"


async def _fetch_file(job: DownloadJob, url: str, dest: Path, expected: int,
                      base_done: int, clock: Dict[str, float]) -> bool:
    """Download one file with resume. Returns False if cancelled/failed (job updated)."""
    tmp = dest.with_name(dest.name + ".tmp")
    if dest.exists() and (not expected or dest.stat().st_size == expected):
        return True
    done = tmp.stat().st_size if tmp.exists() else 0
    if expected and done > expected:
        tmp.unlink()
        done = 0

    for attempt in range(1, MAX_RETRIES + 1):
        if _cancel.get(job.key):
            job.status = "cancelled"
            return False
        job.attempt = attempt
        if attempt > 1:
            wait = min(2 ** (attempt - 2), 60)   # 1s, 2s, 4s … up to 60s
            logger.info(f"Retry {attempt}/{MAX_RETRIES} for {dest.name} in {wait}s …")
            await asyncio.sleep(wait)

        headers = {"Range": f"bytes={done}-"} if done > 0 else {}
        try:
            # httpx drops the Authorization header on the cross-host redirect to HF's CDN
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True,
                                         headers={**hf_headers(), **headers}) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code == 416 and expected and done >= expected:
                        break  # already complete
                    if resp.status_code in (401, 403):
                        job.status, job.error = "failed", ("Access denied — this model is gated. Accept its "
                                                           "license on HuggingFace and add your token in Settings.")
                        return False
                    if resp.status_code not in (200, 206):
                        job.status, job.error = "failed", f"HTTP {resp.status_code} for {dest.name}"
                        logger.error(f"Download {dest.name} got HTTP {resp.status_code}")
                        return False
                    if resp.status_code == 200 and done > 0:
                        logger.warning(f"{dest.name}: server does not support resume, restarting")
                        done = 0

                    job.status = "downloading"
                    with open(tmp, "ab" if done > 0 else "wb") as f:
                        async for chunk in resp.aiter_bytes(CHUNK_SIZE):
                            if _cancel.get(job.key):
                                job.status = "cancelled"
                                return False
                            f.write(chunk)
                            done += len(chunk)
                            clock["session_bytes"] += len(chunk)
                            elapsed = time.time() - clock["start"]
                            speed = clock["session_bytes"] / elapsed if elapsed > 0 else 0
                            job.downloaded_bytes = base_done + done
                            if job.total_bytes:
                                job.pct = round(min(100.0, job.downloaded_bytes / job.total_bytes * 100), 1)
                                job.eta_s = int((job.total_bytes - job.downloaded_bytes) / speed) if speed else 0
                            job.speed_mbs = round(speed / (1024 * 1024), 2)
            if expected and done < expected:
                raise httpx.ReadError(f"stream ended early ({done}/{expected} bytes)")
            break
        except OSError as e:
            if e.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", -1)):
                job.status, job.error = "failed", "The disk is full — free up space and download again (progress is kept)."
                logger.error(f"Disk full while downloading {dest.name}")
                return False
            logger.warning(f"Download interrupted ({dest.name}) attempt {attempt}/{MAX_RETRIES}: {e}")
            job.error = str(e)
            if attempt == MAX_RETRIES:
                job.status = "failed"
                return False
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError,
                httpx.TimeoutException) as e:
            logger.warning(f"Download interrupted ({dest.name}) attempt {attempt}/{MAX_RETRIES}: {e}")
            job.error = str(e)
            if attempt == MAX_RETRIES:
                job.status = "failed"
                logger.error(f"Download permanently failed after {MAX_RETRIES} attempts: {dest.name}")
                return False
        except Exception as e:
            job.status, job.error = "failed", str(e)
            logger.error(f"Download failed {dest.name}: {e}")
            return False

    tmp.replace(dest)
    return True


async def _run_download(job: DownloadJob, files: List[Dict[str, Any]], folder: Path):
    clock = {"start": time.time(), "session_bytes": 0}
    base = 0
    for i, f in enumerate(files, 1):
        job.file_index = i
        dest = folder / f["path"].split("/")[-1]
        ok = await _fetch_file(job, _hf_url(job.model_id, f["path"]), dest, f["size"], base, clock)
        if not ok:
            if job.status == "cancelled":
                _cleanup_partial(files, folder)
            return
        base += f["size"] or dest.stat().st_size

    job.status = "done"
    job.pct = 100.0
    job.downloaded_bytes = job.total_bytes = max(job.total_bytes, base)
    job.eta_s = 0
    _update_manifest(job.filename, {"model_id": job.model_id, "quant": job.quant,
                                    "total_bytes": sum(f["size"] for f in files
                                                       if "mmproj" not in f["path"].lower()),
                                    "downloaded": time.strftime("%Y-%m-%d %H:%M")})
    logger.info(f"Download complete: {job.filename} ({format_bytes(job.total_bytes)})")


def _cleanup_partial(files: List[Dict[str, Any]], folder: Path):
    for f in files:
        tmp = folder / (f["path"].split("/")[-1] + ".tmp")
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def resolve_download(model_id: str, quant: str) -> Dict[str, Any]:
    """Files to fetch for (model, quant). Blocking (HF API call) — run in a thread."""
    if not REPO_ID_RE.match(model_id):
        raise ValueError("Invalid model id")
    groups = quant_files(model_id)
    group = next((g for g in groups if g["quant"].upper() == quant.upper()), None)
    if group is None:
        avail = ", ".join(g["quant"] for g in groups) or "none"
        raise LookupError(f"Quant {quant} not found in {model_id} (available: {avail})")
    files = list(group["files"])
    entry = find_model(model_id)
    needs_mmproj = entry.is_vision if entry else False
    if needs_mmproj or entry is None:
        mm = mmproj_file(model_id)
        if mm:
            files.append(mm)
    folder = _local_folder(model_id)
    primary = folder / group["files"][0]["path"].split("/")[-1]

    # Disk space: what is still missing (already-complete files and partial .tmp count as done)
    need = 0
    for f in files:
        dest = folder / f["path"].split("/")[-1]
        tmp = dest.with_name(dest.name + ".tmp")
        have = dest.stat().st_size if dest.exists() else (tmp.stat().st_size if tmp.exists() else 0)
        need += max(0, f["size"] - have)
    free = shutil.disk_usage(MODELS_DIR).free
    if need + DISK_RESERVE > free:
        raise DiskSpaceError(f"Not enough disk space: this download needs {need / 1024 ** 3:.1f} GB "
                             f"but only {free / 1024 ** 3:.1f} GB is free on the drive holding "
                             f"{MODELS_DIR} (1 GB is kept free for the system).")
    return {"files": files, "folder": folder, "primary": primary, "quant": group["quant"], "need": need}


def start_download(model_id: str, quant: str, plan: Dict[str, Any]) -> DownloadJob:
    """Must be called from the event loop (spawns an asyncio task)."""
    key = f"{model_id}:{plan['quant']}"
    existing = _jobs.get(key)
    if existing and existing.status in ("queued", "downloading"):
        return existing
    folder: Path = plan["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    job = DownloadJob(
        key=key, model_id=model_id, quant=plan["quant"],
        filename=rel_model_path(plan["primary"]),
        status="queued", file_count=len(plan["files"]),
        total_bytes=sum(f["size"] for f in plan["files"]),
    )
    _jobs[key] = job
    _cancel[key] = False
    task = asyncio.create_task(_run_download(job, plan["files"], folder))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return job
