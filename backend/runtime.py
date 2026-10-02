"""
llama.cpp runtime manager.

Finds an existing `llama-server` (./bin or PATH) or downloads the matching prebuilt
release from github.com/ggml-org/llama.cpp into ./bin/llama.cpp:
  Windows x64  → CUDA build for NVIDIA GPUs whose driver supports it (fastest), else
                 Vulkan (AMD / Intel / older NVIDIA drivers), else CPU
  Windows arm64→ CPU build
  macOS        → Metal build (arm64) / x64 build
  Linux x64    → CUDA build for NVIDIA, else Vulkan, else CPU
It also tracks the latest llama.cpp release so the UI can offer updates (new model
architectures usually need a recent llama.cpp).
"""
import asyncio
import os
import platform
import re
import shutil
import stat
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import httpx
from pydantic import BaseModel
from .utils import BIN_DIR, get_logger

logger = get_logger("runtime")

RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
INSTALL_DIR = BIN_DIR / "llama.cpp"
VERSION_FILE = INSTALL_DIR / "VERSION"
SERVER_NAMES = ["llama-server.exe", "llama-server"] if os.name == "nt" else ["llama-server"]
MIN_FREE_BYTES = 2 * 1024 ** 3  # CUDA builds + CUDA runtime unpack to ~1 GB
RELEASE_CHECK_TTL = 6 * 3600

# Minimum NVIDIA driver for each CUDA major version (CUDA minor-version compatibility)
_CUDA_MIN_DRIVER = {"win": {12: 528.33, 13: 580.0}, "linux": {12: 525.60, 13: 580.65}}


class RuntimeStatus(BaseModel):
    installed: bool
    path: str = ""
    version: str = ""          # e.g. b11320
    variant: str = ""          # installed build, e.g. win-vulkan-x64
    source: str = ""           # bundled | path | ""
    recommended: str = ""      # best build for this machine, e.g. win-cuda-13.4-x64
    better_build: bool = False # installed build is slower than the recommended one
    latest_version: str = ""
    update_available: bool = False
    installing: bool = False
    pct: float = 0.0
    stage: str = ""            # downloading | extracting | done | failed
    error: str = ""


_state = {"installing": False, "pct": 0.0, "stage": "", "error": ""}
_tasks: Set[asyncio.Task] = set()
_releases: Dict = {"ts": 0.0, "data": None}
_releases_lock = threading.Lock()


# ─── Discovery ─────────────────────────────────────────────────────────────────

def find_server() -> Optional[str]:
    """Bundled install first (known-good), then anything on PATH."""
    if BIN_DIR.exists():
        for name in SERVER_NAMES:
            hits = sorted(p for p in BIN_DIR.rglob(name) if "_llama-staging" not in p.parts)
            if hits:
                return str(hits[0])
    found = shutil.which("llama-server")
    return found or None


def _installed_meta() -> Tuple[str, str]:
    """(version tag, variant) from the VERSION file written at install time."""
    try:
        txt = VERSION_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return "", ""
    m = re.match(r"(b\d+)\s*\(([^)]+)\)", txt)
    if not m:
        return txt, ""
    variant = re.sub(r"\.(zip|tar\.gz)$", "", m.group(2))
    return m.group(1), variant


def _driver_major_ok(driver: str, cuda_major: int, os_key: str) -> bool:
    try:
        v = float(".".join(driver.split(".")[:2]))
    except ValueError:
        return False
    need = _CUDA_MIN_DRIVER[os_key].get(cuda_major)
    return need is not None and v >= need


def candidate_variants(gpu_vendor: str = "", driver: str = "",
                       system: Optional[str] = None, machine: Optional[str] = None) -> List[str]:
    """Build variants to try for this machine, best first. CUDA entries are patterns
    ('win-cuda-*-x64') resolved against the actual release assets."""
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    arm = machine in ("arm64", "aarch64")
    has_gpu = gpu_vendor in ("NVIDIA", "AMD", "Intel")
    if system == "Windows":
        if arm:
            return ["win-cpu-arm64"]
        out = []
        if gpu_vendor == "NVIDIA" and driver:
            out.append("win-cuda-*-x64")
        if has_gpu:
            out.append("win-vulkan-x64")
        return out + ["win-cpu-x64"]
    if system == "Darwin":
        return ["macos-arm64"] if arm else ["macos-x64"]
    if system == "Linux":
        if arm:
            return ["ubuntu-arm64"]
        out = []
        if gpu_vendor == "NVIDIA" and driver:
            out.append("ubuntu-cuda-*-x64")
        if has_gpu:
            out.append("ubuntu-vulkan-x64")
        return out + ["ubuntu-x64"]
    return []


def asset_suffix(gpu_vendor: str = "", integrated: bool = False) -> Optional[str]:
    """Backwards-compatible: first non-CUDA candidate with its archive extension."""
    for v in candidate_variants(gpu_vendor):
        if "*" not in v:
            return v + (".zip" if v.startswith("win") else ".tar.gz")
    return None


# ─── Release lookup ────────────────────────────────────────────────────────────

def _fetch_releases() -> List[Dict]:
    with httpx.Client(timeout=20, follow_redirects=True,
                      headers={"Accept": "application/vnd.github+json"}) as c:
        r = c.get(RELEASES_API, params={"per_page": 8})
        r.raise_for_status()
        return r.json()


def releases(max_age: float = RELEASE_CHECK_TTL) -> Optional[List[Dict]]:
    with _releases_lock:
        if _releases["data"] is not None and time.time() - _releases["ts"] < max_age:
            return _releases["data"]
    try:
        data = _fetch_releases()
    except Exception as e:
        logger.warning(f"Could not check llama.cpp releases: {e}")
        with _releases_lock:
            _releases["ts"] = time.time() - max_age + 900  # retry in 15 min
            return _releases["data"]
    with _releases_lock:
        _releases.update(ts=time.time(), data=data)
    return data


def resolve_variant(variant: str, rels: List[Dict], driver: str = "") -> Optional[Dict]:
    """Find the newest release asset for a variant (pattern). Returns
    {tag, variant, url, size, extra: [(url, size)]} — extra holds the CUDA runtime archive."""
    os_key = "win" if variant.startswith("win") else "linux"
    for rel in rels:
        assets = rel.get("assets", [])
        names = {a["name"]: a for a in assets}
        if "*" in variant:
            prefix, suffix = variant.split("*")
            pat = re.compile(r"^llama-(b\d+)-bin-" + re.escape(prefix) + r"(\d+)\.(\d+)" + re.escape(suffix)
                             + r"\.(zip|tar\.gz)$")
            found = []
            for n in names:
                m = pat.match(n)
                if m and _driver_major_ok(driver, int(m.group(2)), os_key):
                    found.append((int(m.group(2)), int(m.group(3)), m, n))
            if not found:
                continue
            major, minor, m, n = max(found)
            tag, ver = m.group(1), f"{major}.{minor}"
            concrete = f"{prefix}{ver}{suffix}"
            extra = []
            for cand in (f"cudart-llama-bin-{concrete}.{m.group(4)}",
                         f"cudart-llama-{tag}-bin-{concrete}.{m.group(4)}"):
                if cand in names:
                    extra.append((names[cand]["browser_download_url"], int(names[cand].get("size") or 0)))
                    break
            return {"tag": tag, "variant": concrete, "url": names[n]["browser_download_url"],
                    "size": int(names[n].get("size") or 0), "extra": extra}
        pat = re.compile(r"^llama-(b\d+)-bin-" + re.escape(variant) + r"\.(zip|tar\.gz)$")
        for n, a in names.items():
            m = pat.match(n)
            if m:
                return {"tag": m.group(1), "variant": variant, "url": a["browser_download_url"],
                        "size": int(a.get("size") or 0), "extra": []}
    return None


def best_asset(gpu_vendor: str, driver: str, rels: List[Dict], prefer: Optional[str] = None) -> Optional[Dict]:
    cands = candidate_variants(gpu_vendor, driver)
    if prefer:  # explicit choice, e.g. falling back from CUDA to "win-vulkan-x64"
        cands = [prefer] + [c for c in cands if c != prefer]
    for v in cands:
        a = resolve_variant(v, rels, driver)
        if a:
            return a
    return None


_bg_check = {"running": False}


def releases_nonblocking() -> Optional[List[Dict]]:
    """Cached release list; refreshes in a background thread when stale (never blocks)."""
    with _releases_lock:
        fresh = _releases["data"] is not None and time.time() - _releases["ts"] < RELEASE_CHECK_TTL
        data = _releases["data"]
    if not fresh and not _bg_check["running"]:
        _bg_check["running"] = True

        def run():
            try:
                releases()
            finally:
                _bg_check["running"] = False
        threading.Thread(target=run, daemon=True).start()
    return data


def _tag_num(tag: str) -> int:
    m = re.match(r"b(\d+)", tag or "")
    return int(m.group(1)) if m else 0


def status(gpu_vendor: str = "", integrated: bool = False, driver: str = "",
           check_updates: bool = True) -> RuntimeStatus:
    path = find_server()
    version = variant = source = ""
    if path:
        bundled = Path(path).resolve().is_relative_to(BIN_DIR.resolve())
        source = "bundled" if bundled else "path"
        if bundled:
            version, variant = _installed_meta()
    rels = releases_nonblocking() if check_updates else _releases["data"]
    best = best_asset(gpu_vendor, driver, rels) if rels else None
    recommended = best["variant"] if best else (candidate_variants(gpu_vendor, driver) or [""])[0].replace("*", "")
    latest = best["tag"] if best else ""
    better = bool(source == "bundled" and variant and best and
                  variant.split("-")[1] != best["variant"].split("-")[1])  # e.g. vulkan vs cuda
    return RuntimeStatus(
        installed=bool(path), path=path or "", version=version, variant=variant, source=source,
        recommended=recommended, better_build=better, latest_version=latest,
        update_available=bool(source == "bundled" and version and latest and _tag_num(latest) > _tag_num(version)),
        installing=_state["installing"], pct=_state["pct"], stage=_state["stage"], error=_state["error"],
    )


# ─── Installation ──────────────────────────────────────────────────────────────

def _safe_extract(archive: Path, dest: Path):
    dest_r = dest.resolve()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                if not (dest / n).resolve().is_relative_to(dest_r):
                    raise ValueError(f"Unsafe path in archive: {n}")
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            try:
                t.extractall(dest, filter="data")
            except TypeError:  # Python < 3.12
                for mbr in t.getmembers():
                    if not (dest / mbr.name).resolve().is_relative_to(dest_r):
                        raise ValueError(f"Unsafe path in archive: {mbr.name}")
                t.extractall(dest)
    if os.name != "nt":
        for f in dest.rglob("*"):
            if f.is_file() and (f.name.startswith("llama-") or f.suffix in (".so", ".dylib")):
                f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _flatten_cudart(staging: Path):
    """Put CUDA runtime DLLs/.so next to llama-server so the loader finds them."""
    server = next(staging.rglob(SERVER_NAMES[0]), None)
    if not server:
        return
    for f in staging.rglob("*"):
        if f.is_file() and re.match(r"(cudart|cublas|cublasLt)", f.name, re.I) and f.parent != server.parent:
            shutil.copy2(f, server.parent / f.name)


async def _download(c: httpx.AsyncClient, url: str, dest: Path, base: int, total: int):
    done = 0
    async with c.stream("GET", url) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            async for chunk in resp.aiter_bytes(1024 * 256):
                f.write(chunk)
                done += len(chunk)
                if total:
                    _state["pct"] = round(min(95.0, (base + done) / total * 95), 1)
    return done


async def _install(gpu_vendor: str, driver: str, prefer: Optional[str]):
    staging = BIN_DIR / "_llama-staging"
    archives: List[Path] = []
    try:
        _state.update(installing=True, pct=0.0, stage="downloading", error="")
        rels = await asyncio.to_thread(releases, 0)
        if not rels:
            raise ConnectionError("Could not reach GitHub to download llama.cpp")
        asset = best_asset(gpu_vendor, driver, rels, prefer)
        if not asset:
            raise LookupError(f"No prebuilt llama.cpp for {platform.system()} {platform.machine()}")
        need = asset["size"] + sum(s for _, s in asset["extra"])
        if shutil.disk_usage(BIN_DIR).free < need * 3 + MIN_FREE_BYTES // 4:
            raise OSError(f"Not enough free disk space to install llama.cpp (needs about "
                          f"{(need * 3) / 1024 ** 3:.1f} GB free)")
        logger.info(f"Installing llama.cpp {asset['tag']} ({asset['variant']})")

        base = 0
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=True) as c:
            for i, (url, size) in enumerate([(asset["url"], asset["size"])] + asset["extra"]):
                ext = ".zip" if url.endswith(".zip") else ".tar.gz"
                a = BIN_DIR / f"_llama-download-{i}{ext}"
                archives.append(a)
                base += await _download(c, url, a, base, need)

        _state["stage"] = "extracting"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        for a in archives:
            await asyncio.to_thread(_safe_extract, a, staging)
        if not any(staging.rglob(SERVER_NAMES[0])):
            raise FileNotFoundError("llama-server was not found in the downloaded archive")
        if asset["extra"]:
            _flatten_cudart(staging)
        (staging / "VERSION").write_text(f"{asset['tag']} ({asset['variant']})", encoding="utf-8")
        if INSTALL_DIR.exists():
            shutil.rmtree(INSTALL_DIR, ignore_errors=True)
        staging.rename(INSTALL_DIR)
        _state.update(pct=100.0, stage="done")
        logger.info(f"llama.cpp {asset['tag']} ({asset['variant']}) installed to {INSTALL_DIR}")
    except Exception as e:
        logger.error(f"llama.cpp install failed: {e}")
        _state.update(stage="failed", error=str(e) or e.__class__.__name__)
    finally:
        _state["installing"] = False
        for a in archives:
            a.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def start_install(gpu_vendor: str = "", integrated: bool = False, driver: str = "",
                  prefer: Optional[str] = None) -> RuntimeStatus:
    """Kick off a background install (also used for updates / switching builds).
    Must be called from the event loop."""
    if not _state["installing"]:
        if not candidate_variants(gpu_vendor, driver):
            _state.update(stage="failed", error=f"No prebuilt llama.cpp for {platform.system()} {platform.machine()}")
        else:
            _state.update(installing=True, pct=0.0, stage="downloading", error="")
            task = asyncio.create_task(_install(gpu_vendor, driver, prefer))
            _tasks.add(task)
            task.add_done_callback(_tasks.discard)
    return status(gpu_vendor, integrated, driver, check_updates=False)
