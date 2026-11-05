"""
Hardware knowledge for speed estimates and simulation.

- GPU_BANDWIDTH: published peak memory bandwidth of common GPUs and Apple chips. Token
  generation is memory-bound, so this is what sets tok/s; it replaces the old guess by
  VRAM size whenever the detected GPU name matches.
- PROFILES: whole machines to score models against ("what would run on an RTX 4090
  desktop / an M4 Max?"), in the spirit of llmfit's hardware profiles (MIT,
  github.com/AlexsJones/llmfit — the RTX 4090, M3 Max and Ryzen AI Max+ 395 figures
  come from its bundled profiles).
"""
import re
from typing import Dict, List, Optional, Tuple

from .hardware import GPUInfo, HardwareProfile

# (name fragment, GB/s). Matched case-insensitively against the GPU name; the longest
# matching fragment wins, so "RTX 4070 Ti Super" beats "RTX 4070".
GPU_BANDWIDTH: List[Tuple[str, float]] = [
    # NVIDIA GeForce
    ("RTX 5090", 1792), ("RTX 5080", 960), ("RTX 5070 Ti", 896), ("RTX 5070", 672),
    ("RTX 5060 Ti", 448), ("RTX 5060", 448),
    ("RTX 4090", 1008), ("RTX 4080 Super", 736), ("RTX 4080", 717), ("RTX 4070 Ti Super", 672),
    ("RTX 4070 Ti", 504), ("RTX 4070 Super", 504), ("RTX 4070", 504), ("RTX 4060 Ti", 288),
    ("RTX 4060", 272),
    ("RTX 3090 Ti", 1008), ("RTX 3090", 936), ("RTX 3080 Ti", 912), ("RTX 3080", 760),
    ("RTX 3070 Ti", 608), ("RTX 3070", 448), ("RTX 3060 Ti", 448), ("RTX 3060", 360),
    ("RTX 3050", 224), ("RTX 2080 Ti", 616), ("RTX 2080", 448), ("RTX 2070", 448),
    ("RTX 2060", 336), ("GTX 1660", 192), ("GTX 1080 Ti", 484), ("GTX 1080", 320),
    # NVIDIA workstation / datacenter
    ("RTX PRO 6000", 1792), ("RTX 6000 Ada", 960), ("RTX A6000", 768), ("RTX A5000", 768),
    ("RTX A4000", 448), ("L40S", 864), ("A100", 1555), ("H100", 3350), ("H200", 4800),
    ("GB10", 273), ("DGX Spark", 273),
    # AMD Radeon / Instinct
    ("RX 9070 XT", 640), ("RX 9070", 640), ("RX 7900 XTX", 960), ("RX 7900 XT", 800),
    ("RX 7900 GRE", 576), ("RX 7800 XT", 624), ("RX 7700 XT", 432), ("RX 7600", 288),
    ("RX 6950 XT", 576), ("RX 6900 XT", 512), ("RX 6800 XT", 512), ("RX 6800", 512),
    ("RX 6700 XT", 384), ("Radeon 8060S", 256), ("MI300X", 5300), ("MI250", 3277),
    # Intel Arc
    ("Arc A770", 560), ("Arc A750", 512), ("Arc B580", 456), ("Arc B570", 380),
    # Apple Silicon (unified memory)
    ("M1 Ultra", 800), ("M1 Max", 400), ("M1 Pro", 200), ("M1", 68),
    ("M2 Ultra", 800), ("M2 Max", 400), ("M2 Pro", 200), ("M2", 100),
    ("M3 Ultra", 819), ("M3 Max", 400), ("M3 Pro", 150), ("M3", 100),
    ("M4 Max", 546), ("M4 Pro", 273), ("M4", 120),
]


def gpu_bandwidth(name: str) -> Optional[float]:
    """Peak memory bandwidth (GB/s) for a known GPU name, else None."""
    best: Optional[Tuple[int, float]] = None
    for frag, bw in GPU_BANDWIDTH:
        if re.search(r"(?<![\w])" + re.escape(frag) + r"(?![\w])", name, re.I):
            if best is None or len(frag) > best[0]:
                best = (len(frag), bw)
    return best[1] if best else None


# Whole machines. vram = dedicated GPU memory (0 = shared / unified memory).
PROFILES: List[Dict] = [
    {"id": "laptop-16gb", "label": "Laptop, 16 GB, integrated graphics", "cpu": "8-core laptop CPU",
     "cores": 8, "ram": 16, "gpu": "Integrated graphics", "vendor": "Intel", "vram": 0,
     "gpu_bw": 68, "cpu_bw": 68},
    {"id": "rtx-3060-12gb", "label": "Desktop, RTX 3060 12 GB, 32 GB RAM", "cpu": "8-core desktop CPU",
     "cores": 8, "ram": 32, "gpu": "NVIDIA GeForce RTX 3060", "vendor": "NVIDIA", "vram": 12},
    {"id": "rtx-4060-8gb", "label": "Desktop, RTX 4060 8 GB, 32 GB RAM", "cpu": "8-core desktop CPU",
     "cores": 8, "ram": 32, "gpu": "NVIDIA GeForce RTX 4060", "vendor": "NVIDIA", "vram": 8},
    {"id": "rtx-4070-12gb", "label": "Desktop, RTX 4070 12 GB, 32 GB RAM", "cpu": "8-core desktop CPU",
     "cores": 8, "ram": 32, "gpu": "NVIDIA GeForce RTX 4070", "vendor": "NVIDIA", "vram": 12},
    {"id": "rtx-3090-24gb", "label": "Desktop, RTX 3090 24 GB, 64 GB RAM", "cpu": "12-core desktop CPU",
     "cores": 12, "ram": 64, "gpu": "NVIDIA GeForce RTX 3090", "vendor": "NVIDIA", "vram": 24},
    {"id": "rtx-4090-24gb", "label": "Desktop, RTX 4090 24 GB, 64 GB RAM", "cpu": "16-core desktop CPU",
     "cores": 16, "ram": 64, "gpu": "NVIDIA GeForce RTX 4090", "vendor": "NVIDIA", "vram": 24,
     "cpu_bw": 89.6},
    {"id": "rtx-5090-32gb", "label": "Desktop, RTX 5090 32 GB, 96 GB RAM", "cpu": "16-core desktop CPU",
     "cores": 16, "ram": 96, "gpu": "NVIDIA GeForce RTX 5090", "vendor": "NVIDIA", "vram": 32,
     "cpu_bw": 89.6},
    {"id": "rx-7900-xtx-24gb", "label": "Desktop, Radeon RX 7900 XTX 24 GB, 64 GB RAM",
     "cpu": "12-core desktop CPU", "cores": 12, "ram": 64, "gpu": "AMD Radeon RX 7900 XTX",
     "vendor": "AMD", "vram": 24},
    {"id": "apple-m1-16gb", "label": "MacBook Air, M1, 16 GB", "cpu": "Apple M1", "cores": 8,
     "ram": 16, "gpu": "Apple M1 GPU", "vendor": "Apple", "vram": 0},
    {"id": "apple-m2-pro-32gb", "label": "MacBook Pro, M2 Pro, 32 GB", "cpu": "Apple M2 Pro", "cores": 12,
     "ram": 32, "gpu": "Apple M2 Pro GPU", "vendor": "Apple", "vram": 0},
    {"id": "apple-m3-max-128gb", "label": "MacBook Pro, M3 Max, 128 GB", "cpu": "Apple M3 Max", "cores": 16,
     "ram": 128, "gpu": "Apple M3 Max GPU", "vendor": "Apple", "vram": 0},
    {"id": "apple-m4-max-128gb", "label": "MacBook Pro, M4 Max, 128 GB", "cpu": "Apple M4 Max", "cores": 16,
     "ram": 128, "gpu": "Apple M4 Max GPU", "vendor": "Apple", "vram": 0},
    {"id": "ryzen-ai-max-plus-395", "label": "Ryzen AI Max+ 395, 128 GB unified", "cpu": "AMD Ryzen AI Max+ 395",
     "cores": 16, "ram": 128, "gpu": "AMD Radeon 8060S", "vendor": "AMD", "vram": 96, "unified": True,
     "cpu_bw": 256},
    {"id": "dgx-spark", "label": "NVIDIA DGX Spark, 128 GB unified", "cpu": "NVIDIA GB10 (20-core Arm)",
     "cores": 20, "ram": 128, "gpu": "NVIDIA GB10", "vendor": "NVIDIA", "vram": 120, "unified": True,
     "cpu_bw": 273},
]


def list_profiles() -> List[Dict]:
    return [{"id": p["id"], "label": p["label"], "ram_gb": p["ram"], "vram_gb": p["vram"],
             "gpu": p["gpu"], "cores": p["cores"], "unified": p["vendor"] == "Apple" or bool(p.get("unified")),
             "gpu_bandwidth_gbps": p.get("gpu_bw") or gpu_bandwidth(p["gpu"])} for p in PROFILES]


def profile_hardware_for(profile_id: str, os_name: str = "") -> HardwareProfile:
    """A HardwareProfile describing one of the bundled machines, as if freshly booted."""
    p = next((x for x in PROFILES if x["id"] == profile_id), None)
    if p is None:
        raise ValueError(f"Unknown hardware profile: {profile_id}")
    ram = float(p["ram"])
    avail = round(ram * 0.85, 1)
    apple = p["vendor"] == "Apple"
    unified = apple or bool(p.get("unified"))
    if apple:
        vram, free = round(ram * 0.75, 1), round(avail * 0.75, 1)
    elif p["vram"]:
        vram = float(p["vram"])
        free = round(vram * 0.95, 1) if not unified else round(min(vram, avail), 1)
    else:
        vram, free = 2.0, 2.0
    gpu = GPUInfo(name=p["gpu"], vendor=p["vendor"], total_vram_gb=vram, free_vram_gb=free,
                  cuda=p["vendor"] == "NVIDIA", metal=apple, vulkan=not apple,
                  integrated=unified or not p["vram"])
    return HardwareProfile(
        os_name=os_name or ("Darwin" if apple else "Linux"), os_version="", cpu_name=p["cpu"],
        physical_cores=p["cores"], logical_cores=p["cores"] * (1 if apple else 2), cpu_freq_ghz=3.5,
        total_ram_gb=ram, available_ram_gb=avail, gpu=gpu, score=0.0,
        gpu_bandwidth_gbps=p.get("gpu_bw") or gpu_bandwidth(p["gpu"]), cpu_bandwidth_gbps=p.get("cpu_bw"),
        simulated=p["label"],
    )
