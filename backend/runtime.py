"""
llama.cpp runtime manager.

Finds an existing `llama-server` (./bin or PATH) or downloads the matching prebuilt
release from github.com/ggml-org/llama.cpp into ./bin/llama.cpp:
  Windows x64  → Vulkan build (NVIDIA / AMD / Intel GPUs) or CPU build when no GPU
  Windows arm64→ CPU build
  macOS        → Metal build (arm64) / x64 build
  Linux x64    → Vulkan build or CPU build
"""
import asyncio
import os
import platform
import re
import shutil
import stat
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Optional, Set
import httpx
from pydantic import BaseModel
from .utils import BIN_DIR, get_logger

logger = get_logger("runtime")

RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
INSTALL_DIR = BIN_DIR / "llama.cpp"
VERSION_FILE = INSTALL_DIR / "VERSION"
SERVER_NAMES = ["llama-server.exe", "llama-server"] if os.name == "nt" else ["llama-server"]


class RuntimeStatus(BaseModel):
    installed: bool
    path: str = ""
    version: str = ""
    source: str = ""           # bundled | path | ""
    variant: str = ""          # asset suffix that would be / was installed
    installing: bool = False
    pct: float = 0.0
    stage: str = ""            # downloading | extracting | done | failed
    error: str = ""


_state = {"installing": False, "pct": 0.0, "stage": "", "error": ""}
_tasks: Set[asyncio.Task] = set()


def find_server() -> Optional[str]:
    """Bundled install first (known-good), then anything on PATH."""
    if BIN_DIR.exists():
        for name in SERVER_NAMES:
            hits = sorted(BIN_DIR.rglob(name))
            if hits:
                return str(hits[0])
    for name in ("llama-server",):
        found = shutil.which(name)
        if found:
            return found
    return None


def asset_suffix(gpu_vendor: str = "", integrated: bool = False) -> Optional[str]:
    """Release asset suffix for this platform, e.g. 'win-vulkan-x64.zip'."""
    system = platform.system()
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    has_gpu = gpu_vendor in ("NVIDIA", "AMD", "Intel")
    if system == "Windows":
        if arm:
            return "win-cpu-arm64.zip"
        return "win-vulkan-x64.zip" if has_gpu else "win-cpu-x64.zip"
    if system == "Darwin":
        return "macos-arm64.tar.gz" if arm else "macos-x64.tar.gz"
    if system == "Linux":
        if arm:
            return "ubuntu-arm64.tar.gz"
        return "ubuntu-vulkan-x64.tar.gz" if has_gpu else "ubuntu-x64.tar.gz"
    return None


def status(gpu_vendor: str = "", integrated: bool = False) -> RuntimeStatus:
    path = find_server()
    version = ""
    source = ""
    if path:
        bundled = Path(path).resolve().is_relative_to(BIN_DIR.resolve())
        source = "bundled" if bundled else "path"
        if bundled and VERSION_FILE.exists():
            version = VERSION_FILE.read_text(encoding="utf-8").strip()
    return RuntimeStatus(
        installed=bool(path), path=path or "", version=version, source=source,
        variant=asset_suffix(gpu_vendor, integrated) or "",
        installing=_state["installing"], pct=_state["pct"],
        stage=_state["stage"], error=_state["error"],
    )


async def _latest_asset(suffix: str):
    pat = re.compile(r"^llama-(b\d+)-bin-" + re.escape(suffix) + "$")
    async with httpx.AsyncClient(timeout=20, follow_redirects=True,
                                 headers={"Accept": "application/vnd.github+json"}) as c:
        r = await c.get(RELEASES_API, params={"per_page": 10})
        r.raise_for_status()
        for rel in r.json():
            for a in rel.get("assets", []):
                m = pat.match(a["name"])
                if m:
                    return m.group(1), a["browser_download_url"], int(a.get("size") or 0)
    raise LookupError(f"No llama.cpp release asset found for '{suffix}'")


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


async def _install(suffix: str):
    tmp_archive = BIN_DIR / f"_llama-download-{int(time.time())}{'.zip' if suffix.endswith('.zip') else '.tar.gz'}"
    staging = BIN_DIR / "_llama-staging"
    try:
        _state.update(installing=True, pct=0.0, stage="downloading", error="")
        tag, url, size = await _latest_asset(suffix)
        logger.info(f"Installing llama.cpp {tag} ({suffix})")
        done = 0
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=True) as c:
            async with c.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length") or size or 0)
                with open(tmp_archive, "wb") as f:
                    async for chunk in resp.aiter_bytes(1024 * 256):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            _state["pct"] = round(done / total * 95, 1)

        _state["stage"] = "extracting"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        await asyncio.to_thread(_safe_extract, tmp_archive, staging)
        if not any(staging.rglob(SERVER_NAMES[0])):
            raise FileNotFoundError("llama-server was not found in the downloaded archive")
        if INSTALL_DIR.exists():
            shutil.rmtree(INSTALL_DIR, ignore_errors=True)
        staging.rename(INSTALL_DIR)
        VERSION_FILE.write_text(f"{tag} ({suffix})", encoding="utf-8")
        _state.update(pct=100.0, stage="done")
        logger.info(f"llama.cpp {tag} installed to {INSTALL_DIR}")
    except Exception as e:
        logger.error(f"llama.cpp install failed: {e}")
        _state.update(stage="failed", error=str(e) or e.__class__.__name__)
    finally:
        _state["installing"] = False
        tmp_archive.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def start_install(gpu_vendor: str = "", integrated: bool = False) -> RuntimeStatus:
    """Kick off a background install. Must be called from the event loop."""
    if not _state["installing"]:
        suffix = asset_suffix(gpu_vendor, integrated)
        if not suffix:
            _state.update(stage="failed", error=f"No prebuilt llama.cpp for {platform.system()} {platform.machine()}")
        else:
            _state.update(installing=True, pct=0.0, stage="downloading", error="")
            task = asyncio.create_task(_install(suffix))
            _tasks.add(task)
            task.add_done_callback(_tasks.discard)
    return status(gpu_vendor, integrated)
