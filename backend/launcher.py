import atexit
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional
import httpx
from pydantic import BaseModel
from .runtime import find_server
from . import mlx_runtime
from .mlx_proxy import Gateway
from .utils import LOG_DIR, LLM_PORT, MODELS_DIR, NO_WINDOW, get_logger, safe_model_path, llm_api_key

logger = get_logger("launcher")

LOG_FILE = LOG_DIR / "model-server.log"

_lock = threading.Lock()
_proc: Optional[subprocess.Popen] = None
_log_fh = None
_info: Dict = {}
_gateway: Optional[Gateway] = None


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
    api_key_required: bool = False
    backend: str = "llama.cpp"   # llama.cpp | mlx


def _tail_log(n: int = 25) -> str:
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def stop() -> bool:
    global _proc, _info, _log_fh, _gateway
    with _lock:
        stopped = False
        if _gateway is not None:
            try:
                _gateway.stop()
            except Exception:
                pass
            _gateway = None
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
    if model_path.is_dir() and (model_path / "config.json").exists():
        return _launch_mlx(filename, model_path)
    if not model_path.exists() or model_path.suffix.lower() != ".gguf":
        return LaunchStatus(running=False, message=f"'{filename}' is not downloaded yet — download it first.")

    ctx = max(512, min(int(ctx), 262144))
    threads = max(1, min(int(threads), 256))
    gpu_layers = max(0, min(int(gpu_layers), 999))

    args: List[str] = ["-m", str(model_path), "-c", str(ctx), "-t", str(threads),
                       "-ngl", str(gpu_layers), "--port", str(LLM_PORT), "--host", "127.0.0.1",
                       "--alias", model_path.name]
    # Without a key, any web page open in the browser could call the model server
    key = llm_api_key()
    if key:
        args += ["--api-key", key]
    mmproj = sorted(p for p in model_path.parent.glob("*.gguf") if "mmproj" in p.name.lower())
    if mmproj and model_path.parent != MODELS_DIR.resolve():  # legacy flat files: no pairing
        args += ["--mmproj", str(mmproj[0])]

    exe = find_server()
    if not exe:
        cmd = " ".join(["llama-server"] + [f'"{a}"' if " " in a else ("***" if key and a == key else a)
                                           for a in args])
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

    return _spawn([exe] + args, Path(exe).parent, filename, model_path.name, key, "llama.cpp")


def _spawn(cmd_vec: List[str], cwd: Path, filename: str, display: str, key: str, backend: str,
           env: Optional[Dict[str, str]] = None, extra: Optional[Dict] = None,
           after_start=None) -> LaunchStatus:
    global _proc, _info, _log_fh
    try:
        shown_cmd = " ".join("***" if key and a == key else a for a in cmd_vec)
        with _lock:
            _log_fh = open(LOG_FILE, "w", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(cmd_vec, stdout=_log_fh, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, cwd=str(cwd), env=env,
                                    creationflags=NO_WINDOW,
                                    preexec_fn=None if sys.platform == "win32" else _child_preexec)
            _bind_to_parent(proc)
            # Published together so concurrent status() polls always see a complete record
            _proc = proc
            _info = {
                "running": True,
                "pid": proc.pid,
                "model": filename,
                "cmd": shown_cmd,
                "message": f"Loading {display} (PID {proc.pid})…",
                "api_key_required": bool(key),
                "backend": backend,
                **(extra or {}),
            }
            info = dict(_info)
        if after_start:
            err = after_start()
            if err:
                stop()
                return LaunchStatus(running=False, message=err, backend=backend)
        # Give it a moment to fail fast (bad model file, missing GPU driver, …)
        try:
            proc.wait(timeout=2.0)
            err = _tail_log()
            stop()
            return LaunchStatus(running=False, message=f"{backend} server exited immediately:\n{err}",
                                backend=backend)
        except subprocess.TimeoutExpired:
            pass

        logger.info(f"Launched {display} with {backend} (PID {proc.pid})")
        return LaunchStatus(**{k: v for k, v in info.items() if k in LaunchStatus.model_fields})
    except Exception as e:
        logger.error(f"Launch failed: {e}")
        stop()
        return LaunchStatus(running=False, message=str(e), backend=backend)


def _free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launch_mlx(filename: str, model_dir: Path) -> LaunchStatus:
    """Start mlx_lm.server on a private port behind the authenticating gateway on LLM_PORT."""
    global _gateway, _info
    key = llm_api_key()
    py = mlx_runtime.env_python()
    internal = _free_port()
    args = [str(py), "-m", "mlx_lm.server", "--model", str(model_dir), "--host", "127.0.0.1",
            "--port", str(internal), "--max-tokens", "4096",
            # Browsers may not talk to the private port directly; everything goes through the gateway
            "--allowed-origins", "http://127.0.0.1:1"]
    if not mlx_runtime.is_installed():
        with _lock:
            _info = {"running": False, "pid": None, "model": filename, "dry_run": True, "backend": "mlx",
                     "cmd": "mlx_lm.server " + " ".join(args[3:]),
                     "message": "The MLX runtime is not installed. Install it from the app (Settings)."}
        return LaunchStatus(**_info)
    if _port_in_use(LLM_PORT):
        return LaunchStatus(running=False, backend="mlx", message=(
            f"Port {LLM_PORT} is already in use by another program. Stop it, or set "
            f"AUTOTUNER_LLM_PORT to a free port and restart the app."))

    env = {**os.environ,
           "HF_HUB_OFFLINE": "1",            # never download a model just because a client named it
           "TRANSFORMERS_OFFLINE": "1",
           "HF_HOME": str(mlx_runtime.MLX_DIR / "hf-home"),
           "PYTHONUNBUFFERED": "1"}

    def start_gateway():
        global _gateway
        gw = Gateway(LLM_PORT, f"http://127.0.0.1:{internal}", key)
        if not gw.start():
            return f"Could not open port {LLM_PORT} for the model API."
        with _lock:
            _gateway = gw
        return None

    return _spawn(args, model_dir, filename, model_dir.name, key, "mlx", env=env,
                  extra={"internal_port": internal, "chat_url": ""}, after_start=start_gateway)


def _health(info: Optional[Dict] = None) -> bool:
    try:
        if info and info.get("internal_port"):
            r = httpx.get(f"http://127.0.0.1:{info['internal_port']}/health", timeout=1.0)
            return r.status_code == 200
        key = llm_api_key()
        r = httpx.get(f"http://127.0.0.1:{LLM_PORT}/health", timeout=1.0,
                      headers={"Authorization": f"Bearer {key}"} if key else {})
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
            name = "mlx_lm.server" if info.get("backend") == "mlx" else "llama-server"
            return LaunchStatus(running=False, message=f"{name} stopped (exit code {proc.returncode}).\n{tail}".strip())
        ready = _health(info)
        msg = f"Serving {Path(info.get('model', '')).name} on port {LLM_PORT}" if ready else info.get("message", "")
        fields = {k: v for k, v in info.items() if k in LaunchStatus.model_fields}
        return LaunchStatus(**{"running": True, **fields, "ready": ready, "message": msg})
    if info.get("dry_run"):
        return LaunchStatus(**{k: v for k, v in info.items() if k in LaunchStatus.model_fields})
    return LaunchStatus(running=False, message="No model running.")


atexit.register(stop)
