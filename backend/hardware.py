import json
import platform
import re
import subprocess
import shutil
import threading
import time
import psutil
from typing import Optional, Tuple
from pydantic import BaseModel
from .utils import get_logger, NO_WINDOW

logger = get_logger("hardware")


class GPUInfo(BaseModel):
    name: str
    vendor: str
    total_vram_gb: float
    free_vram_gb: float
    cuda: bool = False
    metal: bool = False
    vulkan: bool = False
    # True when the GPU shares system RAM (iGPU / APU). Its "VRAM" is carved out of
    # system memory, so it must not be added on top of RAM when sizing models.
    integrated: bool = False
    driver_version: str = ""   # NVIDIA driver (decides which CUDA build of llama.cpp can run)


class HardwareProfile(BaseModel):
    os_name: str
    os_version: str
    cpu_name: str
    physical_cores: int
    logical_cores: int
    cpu_freq_ghz: float
    total_ram_gb: float
    available_ram_gb: float
    gpu: GPUInfo
    score: float  # 0–100 system tier


def _run(cmd, timeout: float) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           creationflags=NO_WINDOW)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception as e:
        logger.warning(f"{cmd[0]} failed: {e}")
    return None


def _is_integrated(name: str, vendor: str) -> bool:
    n = name.upper()
    if vendor == "Intel":
        return "ARC" not in n or bool(re.search(r"ARC\(TM\) GRAPHICS|ARC GRAPHICS", n))
    if vendor == "AMD":
        # Discrete Radeons carry RX / PRO W / Instinct / VII branding; APUs don't
        return not re.search(r"\bRX\b|PRO W\d|INSTINCT|RADEON VII|VEGA (56|64)|FRONTIER", n)
    return False


# Windows: Win32_VideoController.AdapterRAM is a uint32 and caps at 4 GB, so read the
# 64-bit qwMemorySize from the display-adapter registry keys as well.
_WIN_GPU_PS = r"""
$ctl = @(Get-CimInstance Win32_VideoController | ForEach-Object { @{ Name = $_.Name; AdapterRAM = [int64]$_.AdapterRAM } })
$reg = @(Get-ItemProperty 'HKLM:\SYSTEM\ControlSet001\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0*' -ErrorAction SilentlyContinue |
  ForEach-Object { @{ Name = $_.DriverDesc; Mem = [int64]$_.'HardwareInformation.qwMemorySize' } })
$cpu = (Get-CimInstance Win32_Processor)[0].Name
@{ ctl = $ctl; reg = $reg; cpu = $cpu } | ConvertTo-Json -Depth 4 -Compress
"""


def _windows_probe() -> Tuple[Optional[dict], Optional[str]]:
    out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _WIN_GPU_PS], timeout=10)
    if not out:
        return None, None
    try:
        data = json.loads(out)
    except Exception as e:
        logger.warning(f"Could not parse Windows hardware probe: {e}")
        return None, None
    cpu = (data.get("cpu") or "").strip() or None
    ctl = [c for c in (data.get("ctl") or []) if c and c.get("Name")]
    if not ctl:
        return None, cpu
    reg_mem = {}
    for r in data.get("reg") or []:
        if r and r.get("Name") and r.get("Mem"):
            reg_mem[r["Name"]] = max(reg_mem.get(r["Name"], 0), int(r["Mem"]))

    def priority(it):
        n = it["Name"].upper()
        if "NVIDIA" in n:
            return 0
        if ("RADEON" in n or "AMD" in n) and not _is_integrated(it["Name"], "AMD"):
            return 1
        if "ARC" in n and not _is_integrated(it["Name"], "Intel"):
            return 1
        if "MICROSOFT" in n or "VIRTUAL" in n or "PARSEC" in n:
            return 9
        return 2

    item = sorted(ctl, key=priority)[0]
    mem = max(int(item.get("AdapterRAM") or 0), reg_mem.get(item["Name"], 0))
    return {"name": item["Name"], "mem_bytes": mem}, cpu


def _detect_gpu(win_gpu: Optional[dict]) -> GPUInfo:
    system = platform.system()
    vm = psutil.virtual_memory()

    # --- NVIDIA via nvidia-smi (exact free/total VRAM) ---
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=gpu_name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits"], timeout=5)
        if out:
            parts = [p.strip() for p in out.split("\n")[0].split(",")]
            if len(parts) >= 3:
                try:
                    return GPUInfo(
                        name=parts[0], vendor="NVIDIA",
                        total_vram_gb=round(int(parts[1]) / 1024, 2),
                        free_vram_gb=round(int(parts[2]) / 1024, 2),
                        cuda=True, vulkan=True,
                        driver_version=parts[3] if len(parts) > 3 else "",
                    )
                except ValueError:
                    pass

    # --- Apple Silicon: unified memory, Metal can use ~75% of it ---
    if system == "Darwin" and platform.machine() == "arm64":
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=3) or "Apple Silicon"
        return GPUInfo(
            name=f"{brand} GPU",
            vendor="Apple",
            total_vram_gb=round(vm.total * 0.75 / 1024 ** 3, 2),
            free_vram_gb=round(vm.available * 0.75 / 1024 ** 3, 2),
            metal=True, integrated=True,
        )

    # --- Windows (CIM + registry) ---
    if win_gpu:
        name = win_gpu["name"]
        up = name.upper()
        vendor = ("NVIDIA" if "NVIDIA" in up
                  else "AMD" if ("AMD" in up or "RADEON" in up)
                  else "Intel" if "INTEL" in up
                  else "Generic")
        integrated = _is_integrated(name, vendor)
        mem = win_gpu["mem_bytes"]
        if mem <= 512 * 1024 ** 2:
            # Unknown / token value: assume shared memory up to 1/8 of RAM
            mem = max(int(vm.total // 8), 2 * 1024 ** 3)
            integrated = True
        total = round(mem / 1024 ** 3, 2)
        free = round(total * 0.9, 2) if not integrated else round(total * vm.available / vm.total, 2)
        return GPUInfo(name=name, vendor=vendor, total_vram_gb=total, free_vram_gb=free,
                       vulkan=True, integrated=integrated)

    # --- Linux: lspci for a name, sysfs for AMD VRAM ---
    if system == "Linux":
        out = _run(["lspci"], timeout=3) or ""
        for line in out.splitlines():
            if re.search(r"VGA|3D controller|Display", line):
                name = line.split(":", 2)[-1].strip()
                up = name.upper()
                vendor = "AMD" if ("AMD" in up or "ATI" in up) else "Intel" if "INTEL" in up else "NVIDIA" if "NVIDIA" in up else "Generic"
                vram = 0
                try:
                    from pathlib import Path
                    for f in Path("/sys/class/drm").glob("card*/device/mem_info_vram_total"):
                        vram = max(vram, int(f.read_text().strip()))
                except Exception:
                    pass
                integrated = _is_integrated(name, vendor) or vram < 1024 ** 3
                if vram < 1024 ** 3:
                    vram = max(int(vm.total // 8), 2 * 1024 ** 3)
                total = round(vram / 1024 ** 3, 2)
                return GPUInfo(name=name, vendor=vendor, total_vram_gb=total,
                               free_vram_gb=round(total * 0.9, 2), vulkan=True, integrated=integrated)

    # --- Fallback: no usable GPU information ---
    return GPUInfo(
        name="Integrated / Unknown GPU", vendor="Generic",
        total_vram_gb=round(vm.total * 0.125 / 1024 ** 3, 2),
        free_vram_gb=round(vm.available * 0.125 / 1024 ** 3, 2),
        integrated=True,
    )


# The slow probes (PowerShell, nvidia-smi) are cached; RAM figures are refreshed on every call.
_STATIC_TTL = 300
_static_lock = threading.Lock()
_static: Optional[dict] = None
_static_ts = 0.0


def _static_profile() -> dict:
    global _static, _static_ts
    with _static_lock:
        if _static is None or time.time() - _static_ts > _STATIC_TTL:
            win_gpu, win_cpu = _windows_probe() if platform.system() == "Windows" else (None, None)
            p_cores = psutil.cpu_count(logical=False) or 4
            cpu_name = win_cpu or platform.processor() or f"{p_cores}-core CPU"
            if platform.system() == "Darwin":
                cpu_name = _run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=3) or cpu_name
            elif platform.system() == "Linux":
                try:
                    for line in open("/proc/cpuinfo", encoding="utf-8"):
                        if line.startswith("model name"):
                            cpu_name = line.split(":", 1)[1].strip()
                            break
                except Exception:
                    pass
            _static = {
                "gpu": _detect_gpu(win_gpu),
                "cpu_name": cpu_name,
                "p_cores": p_cores,
                "l_cores": psutil.cpu_count(logical=True) or p_cores,
            }
            _static_ts = time.time()
        return _static


def profile_hardware() -> HardwareProfile:
    s = _static_profile()
    mem = psutil.virtual_memory()
    freq = psutil.cpu_freq()
    freq_ghz = round((freq.current if freq and freq.current else 2500) / 1000, 2)
    gpu: GPUInfo = s["gpu"].model_copy()
    if gpu.integrated and gpu.vendor != "Apple":
        gpu.free_vram_gb = round(gpu.total_vram_gb * mem.available / mem.total, 2)

    p_cores = s["p_cores"]
    total_ram = round(mem.total / 1024 ** 3, 2)
    avail_ram = round(mem.available / 1024 ** 3, 2)

    # Compute system tier score 0–100
    ram_score = min(25.0, total_ram * 0.9)
    vram_score = min(50.0, gpu.total_vram_gb * (1.5 if gpu.integrated else 3.5))
    cpu_score = min(25.0, p_cores * 2.5)
    score = round(ram_score + vram_score + cpu_score, 1)

    return HardwareProfile(
        os_name=platform.system(),
        os_version=platform.release(),
        cpu_name=s["cpu_name"],
        physical_cores=p_cores,
        logical_cores=s["l_cores"],
        cpu_freq_ghz=freq_ghz,
        total_ram_gb=total_ram,
        available_ram_gb=avail_ram,
        gpu=gpu,
        score=score,
    )
