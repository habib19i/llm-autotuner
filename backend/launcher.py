import atexit
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional
import httpx
from pydantic import BaseModel
from .runtime import find_server
from .utils import LOG_DIR, LLM_PORT, MODELS_DIR, NO_WINDOW, get_logger, safe_model_path

logger = get_logger("launcher")

LOG_FILE = LOG_DIR / "llama-server.log"

_lock = threading.Lock()
_proc: Optional[subprocess.Popen] = None
_log_fh = None
_info: Dict = {}


# ─── Tie llama-server's lifetime to ours ───────────────────────────────────────
# Windows doesn't kill child processes when the parent dies (e.g. the console window is
# closed), so the server is placed in a Job Object that terminates it when our handle closes.

_job = None
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    try:
        _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _k32.CreateJobObjectW.restype = wintypes.HANDLE
        _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        _job = _k32.CreateJobObjectW(None, None)
        _lim = _ExtendedLimits()
        _lim.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(_job, 9, ctypes.byref(_lim), ctypes.sizeof(_lim)):
            _job = None
    except Exception as e:  # pragma: no cover - best effort
        logger.warning(f"Could not create job object: {e}")
        _job = None


def _bind_to_parent(proc: subprocess.Popen):
    if _job is not None:
        try:
            _k32.AssignProcessToJobObject(_job, wintypes.HANDLE(int(proc._handle)))
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not attach llama-server to job object: {e}")


def _child_preexec():  # POSIX: die with the parent (Linux) and ignore terminal Ctrl+C
    import os
    import signal
    os.setsid()
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
        except Exception:
            pass


class LaunchStatus(BaseModel):
    running: bool
    ready: bool = False        # server answered /health — model loaded, API usable
    pid: Optional[int] = None
    model: str = ""
    cmd: str = ""
    message: str = ""
    port: int = LLM_PORT
    base_url: str = f"http://127.0.0.1:{LLM_PORT}/v1"
    chat_url: str = f"http://127.0.0.1:{LLM_PORT}"
    dry_run: bool = False


def _tail_log(n: int = 25) -> str:
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def stop() -> bool:
    global _proc, _info, _log_fh
    with _lock:
        stopped = False
        if _proc and _proc.poll() is None:
            try:
                _proc.terminate()
                _proc.wait(timeout=6)
            except Exception:
                try:
                    _proc.kill()
                    _proc.wait(timeout=3)
                except Exception:
                    pass
            stopped = True
        elif _info.get("dry_run"):
            stopped = True
        _proc = None
        _info = {}
        if _log_fh:
            try:
                _log_fh.close()
            except Exception:
                pass
            _log_fh = None
        return stopped


def _port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def launch(filename: str, ctx: int, threads: int, gpu_layers: int) -> LaunchStatus:
    global _proc, _info, _log_fh
    stop()

    try:
        model_path = safe_model_path(filename)
    except ValueError as e:
        return LaunchStatus(running=False, message=str(e))
    if not model_path.exists() or model_path.suffix.lower() != ".gguf":
        return LaunchStatus(running=False, message=f"'{filename}' is not downloaded yet — download it first.")

    ctx = max(512, min(int(ctx), 262144))
    threads = max(1, min(int(threads), 256))
    gpu_layers = max(0, min(int(gpu_layers), 999))

    args: List[str] = ["-m", str(model_path), "-c", str(ctx), "-t", str(threads),
                       "-ngl", str(gpu_layers), "--port", str(LLM_PORT), "--host", "127.0.0.1",
                       "--alias", model_path.name]
    mmproj = sorted(p for p in model_path.parent.glob("*.gguf") if "mmproj" in p.name.lower())
    if mmproj and model_path.parent != MODELS_DIR.resolve():  # legacy flat files: no pairing
        args += ["--mmproj", str(mmproj[0])]

    exe = find_server()
    if not exe:
        cmd = " ".join(["llama-server"] + [f'"{a}"' if " " in a else a for a in args])
        with _lock:
            _info = {
                "running": False, "pid": None, "model": filename, "cmd": cmd, "dry_run": True,
                "message": "llama.cpp runtime is not installed. Install it from the app, "
                           "or run the command manually.",
            }
        return LaunchStatus(**_info)

    if _port_in_use(LLM_PORT):
        return LaunchStatus(running=False, message=(
            f"Port {LLM_PORT} is already in use by another program. Stop it, or set "
            f"AUTOTUNER_LLM_PORT to a free port and restart the app."))

    cmd_vec = [exe] + args
    try:
        with _lock:
            _log_fh = open(LOG_FILE, "w", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(cmd_vec, stdout=_log_fh, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, cwd=str(Path(exe).parent),
                                    creationflags=NO_WINDOW,
                                    preexec_fn=None if sys.platform == "win32" else _child_preexec)
            _bind_to_parent(proc)
            # Published together so concurrent status() polls always see a complete record
            _proc = proc
            _info = {
                "running": True,
                "pid": proc.pid,
                "model": filename,
                "cmd": " ".join(cmd_vec),
                "message": f"Loading {model_path.name} (PID {proc.pid})…",
            }
            info = dict(_info)
        # Give it a moment to fail fast (bad model file, missing GPU driver, …)
        try:
            proc.wait(timeout=2.0)
            err = _tail_log()
            stop()
            return LaunchStatus(running=False, message=f"llama-server exited immediately:\n{err}")
        except subprocess.TimeoutExpired:
            pass

        logger.info(f"Launched {model_path.name} (PID {proc.pid})")
        return LaunchStatus(**info)
    except Exception as e:
        logger.error(f"Launch failed: {e}")
        stop()
        return LaunchStatus(running=False, message=str(e))


def _health() -> bool:
    try:
        r = httpx.get(f"http://127.0.0.1:{LLM_PORT}/health", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def status() -> LaunchStatus:
    global _proc, _info
    with _lock:
        proc, info = _proc, dict(_info)
    if proc is not None:
        if proc.poll() is not None:
            tail = _tail_log(8)
            stop()
            return LaunchStatus(running=False, message=f"llama-server stopped (exit code {proc.returncode}).\n{tail}".strip())
        ready = _health()
        msg = f"Serving {Path(info.get('model', '')).name} on port {LLM_PORT}" if ready else info.get("message", "")
        return LaunchStatus(**{"running": True, **info, "ready": ready, "message": msg})
    if info.get("dry_run"):
        return LaunchStatus(**info)
    return LaunchStatus(running=False, message="No model running.")


atexit.register(stop)
