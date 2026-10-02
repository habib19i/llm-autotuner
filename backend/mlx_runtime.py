"""
MLX runtime manager (Apple Silicon).

MLX models run with Apple's `mlx-lm` package. To keep the app self-contained, it is installed
into ./bin/mlx without touching the system: we download `uv` (a single static binary from
github.com/astral-sh/uv), let it fetch a private Python, and install mlx-lm into a virtual
environment there.
"""
import asyncio
import os
import platform
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Dict, Optional, Set

import httpx
from pydantic import BaseModel

from .utils import BIN_DIR, NO_WINDOW, get_logger

logger = get_logger("mlx_runtime")

MLX_DIR = BIN_DIR / "mlx"
ENV_DIR = MLX_DIR / "env"
UV_DIR = MLX_DIR / "uv"
VERSION_FILE = MLX_DIR / "VERSION"
PYPI_MLX_LM = "https://pypi.org/pypi/mlx-lm/json"
PYTHON_VERSION = "3.12"


def mlx_supported() -> bool:
    """MLX is offered on Apple Silicon Macs. AUTOTUNER_ENABLE_MLX=1 enables it elsewhere
    (mlx also ships CPU builds for Linux/Windows, useful for development)."""
    if os.environ.get("AUTOTUNER_ENABLE_MLX") == "1":
        return True
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def env_python() -> Path:
    """Python with mlx-lm. AUTOTUNER_MLX_PYTHON points at an existing installation instead."""
    custom = os.environ.get("AUTOTUNER_MLX_PYTHON")
    if custom:
        return Path(custom)
    return ENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def installed_version() -> str:
    if os.environ.get("AUTOTUNER_MLX_PYTHON"):
        return "external"
    try:
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def is_installed() -> bool:
    return env_python().exists() and bool(installed_version())


class MlxStatus(BaseModel):
    supported: bool
    installed: bool
    version: str = ""
    latest_version: str = ""
    update_available: bool = False
    installing: bool = False
    pct: float = 0.0
    stage: str = ""   # downloading | python | packages | done | failed
    error: str = ""


_state: Dict = {"installing": False, "pct": 0.0, "stage": "", "error": ""}
_tasks: Set[asyncio.Task] = set()
_latest = {"ts": 0.0, "version": ""}


def _latest_version() -> str:
    """mlx-lm version on PyPI (cached for 6 hours, never raises)."""
    if time.time() - _latest["ts"] < 6 * 3600:
        return _latest["version"]
    _latest["ts"] = time.time()
    try:
        r = httpx.get(PYPI_MLX_LM, timeout=10)
        _latest["version"] = r.json()["info"]["version"]
    except Exception as e:
        logger.debug(f"PyPI check failed: {e}")
    return _latest["version"]


def _vtuple(v: str):
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except ValueError:
        return (0,)


def status(check_updates: bool = True) -> MlxStatus:
    v = installed_version() if env_python().exists() else ""
    latest = _latest_version() if (check_updates and mlx_supported() and v) else _latest["version"]
    return MlxStatus(
        supported=mlx_supported(), installed=bool(v), version=v, latest_version=latest,
        update_available=bool(v and latest and _vtuple(latest) > _vtuple(v)),
        installing=_state["installing"], pct=_state["pct"], stage=_state["stage"], error=_state["error"],
    )


def _uv_asset() -> Optional[str]:
    system, machine = platform.system(), platform.machine().lower()
    arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
    if system == "Darwin":
        return f"uv-{arch}-apple-darwin.tar.gz"
    if system == "Linux":
        return f"uv-{arch}-unknown-linux-gnu.tar.gz"
    if system == "Windows":
        return f"uv-{arch}-pc-windows-msvc.zip"
    return None


def _uv_exe() -> Optional[Path]:
    name = "uv.exe" if os.name == "nt" else "uv"
    hits = sorted(UV_DIR.rglob(name)) if UV_DIR.exists() else []
    return hits[0] if hits else None


async def _run(*cmd: str, env: Optional[Dict[str, str]] = None, timeout: float = 1800) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, **(env or {})}, creationflags=NO_WINDOW)
    out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    text = out.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} {cmd[1] if len(cmd) > 1 else ''} failed:\n{text[-1500:]}")
    return text


def _uv_env() -> Dict[str, str]:
    # Keep uv's Python downloads and cache inside the app folder
    return {"UV_PYTHON_INSTALL_DIR": str(MLX_DIR / "python"), "UV_CACHE_DIR": str(MLX_DIR / "cache"),
            "UV_NO_CONFIG": "1", "UV_PYTHON_PREFERENCE": "only-managed"}


async def _ensure_uv():
    if _uv_exe():
        return
    asset = _uv_asset()
    if not asset:
        raise LookupError(f"No uv build for {platform.system()} {platform.machine()}")
    # Fixed "latest release" download link: no GitHub API call, so no API rate limit
    url = f"https://github.com/astral-sh/uv/releases/latest/download/{asset}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=True) as c:
        archive = MLX_DIR / asset
        async with c.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length") or 0)
            done = 0
            with open(archive, "wb") as f:
                async for chunk in resp.aiter_bytes(1024 * 256):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        _state["pct"] = round(done / total * 20, 1)
    UV_DIR.mkdir(parents=True, exist_ok=True)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(UV_DIR)
    else:
        with tarfile.open(archive) as t:
            try:
                t.extractall(UV_DIR, filter="data")
            except TypeError:
                t.extractall(UV_DIR)
    archive.unlink(missing_ok=True)
    exe = _uv_exe()
    if not exe:
        raise FileNotFoundError("uv binary missing after extraction")
    if os.name != "nt":
        exe.chmod(0o755)


async def _install(upgrade: bool):
    try:
        _state.update(installing=True, pct=0.0, stage="downloading", error="")
        MLX_DIR.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(MLX_DIR).free
        if free < 2 * 1024 ** 3:
            raise OSError(f"Not enough free disk space to install MLX (needs about 2 GB, {free / 1024 ** 3:.1f} GB free)")
        await _ensure_uv()
        uv = str(_uv_exe())
        env = _uv_env()
        if not env_python().exists():
            _state.update(stage="python", pct=25.0)
            await _run(uv, "venv", str(ENV_DIR), "--python", PYTHON_VERSION, env=env)
        _state.update(stage="packages", pct=55.0)
        args = [uv, "pip", "install", "--python", str(env_python())]
        if upgrade:
            args.append("--upgrade")
        # mlx-lm only depends on mlx on macOS; elsewhere (developer mode) add the CPU build
        packages = {"Darwin": ["mlx-lm"], "Linux": ["mlx-lm", "mlx[cpu]"]}.get(platform.system(), ["mlx-lm", "mlx"])
        await _run(*args, *packages, env=env)
        out = await _run(str(env_python()), "-c", "import importlib.metadata as m; print(m.version('mlx-lm'))")
        VERSION_FILE.write_text(out.strip().splitlines()[-1], encoding="utf-8")
        _state.update(pct=95.0)
        try:
            await _run(uv, "cache", "clean", env=env, timeout=300)  # reclaim the download cache
        except Exception:
            pass
        _state.update(pct=100.0, stage="done")
        logger.info(f"MLX runtime ready (mlx-lm {installed_version()})")
    except Exception as e:
        logger.error(f"MLX install failed: {e}")
        _state.update(stage="failed", error=str(e) or e.__class__.__name__)
    finally:
        _state["installing"] = False


def start_install(upgrade: bool = False) -> MlxStatus:
    """Install (or upgrade) mlx-lm in the background. Must be called from the event loop."""
    if not _state["installing"]:
        if not mlx_supported():
            _state.update(stage="failed", error="MLX needs a Mac with Apple Silicon (M1 or newer)")
        else:
            _state.update(installing=True, pct=0.0, stage="downloading", error="")
            task = asyncio.create_task(_install(upgrade))
            _tasks.add(task)
            task.add_done_callback(_tasks.discard)
    return status(check_updates=False)
