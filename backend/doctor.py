"""
Diagnostic report for bug reports (after llmfit's `doctor`): what the app detected, the raw
output of the GPU tools it relied on, and the state of its runtimes. Secrets are left out.
"""
import platform
import shutil
import subprocess
import sys

from .utils import APP_VERSION, LLM_PORT, NO_WINDOW, MODELS_DIR, get_setting


def _raw(cmd) -> str:
    exe = shutil.which(cmd[0])
    if not exe:
        return f"({cmd[0]} not found)"
    try:
        r = subprocess.run([exe] + cmd[1:], capture_output=True, text=True, timeout=8, creationflags=NO_WINDOW)
        return (r.stdout or r.stderr or "").strip()[:4000] or f"(no output, exit {r.returncode})"
    except Exception as e:
        return f"(failed: {e})"


def report() -> str:
    from .hardware import profile_hardware
    from . import runtime, mlx_runtime, other_runtimes, bench
    from .launcher import status as launch_status
    from .scoring import _bandwidths, efficiency, context_cap
    hw = profile_hardware()
    gpu_bw, cpu_bw = _bandwidths(hw)
    lines = [
        f"LLM Autotuner {APP_VERSION}",
        f"OS: {platform.system()} {platform.release()} ({platform.machine()}), Python {sys.version.split()[0]}",
        "",
        "## Detected hardware",
        hw.model_dump_json(indent=2),
        f"Memory bandwidth used for estimates: GPU {gpu_bw:.0f} GB/s, CPU {cpu_bw:.0f} GB/s",
        f"Speed efficiency: {efficiency()}  Context cap: {context_cap() or 'none'}",
        "",
        "## Raw GPU tool output",
        "$ nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv",
        _raw(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv"]),
    ]
    if platform.system() == "Linux":
        lines += ["$ rocm-smi --showmeminfo vram", _raw(["rocm-smi", "--showmeminfo", "vram"]),
                  "$ lspci (display)", "\n".join(l for l in _raw(["lspci"]).splitlines()
                                                 if any(k in l for k in ("VGA", "3D", "Display")))]
    elif platform.system() == "Darwin":
        lines += ["$ sysctl machdep.cpu.brand_string hw.memsize",
                  _raw(["sysctl", "machdep.cpu.brand_string", "hw.memsize"])]
    rt = runtime.status(hw.gpu.vendor, hw.gpu.integrated, hw.gpu.driver_version, check_updates=False)
    st = launch_status()
    lines += [
        "",
        "## Runtimes",
        f"llama.cpp: installed={rt.installed} source={rt.source} version={rt.version} variant={rt.variant} "
        f"recommended={rt.recommended}",
        f"MLX: supported={mlx_runtime.mlx_supported()}",
        f"Model server: running={st.running} ready={st.ready} backend={st.backend} port={LLM_PORT} model={st.model}",
    ]
    for r in other_runtimes.detect(force=True).values():
        lines.append(f"{r['name']}: {'running' if r['running'] else 'not running'} at {r['url']}, "
                     f"{len(r['models'])} model(s)")
    lines += ["", "## Storage", f"Models folder: {MODELS_DIR}",
              f"Benchmarks on this machine: {len(bench.for_hardware(hw))}",
              f"HuggingFace token set: {bool(get_setting('hf_token'))}"]
    return "\n".join(lines) + "\n"
