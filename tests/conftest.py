import os
import sys
import tempfile
from pathlib import Path

import pytest

# Isolate all user data (models, cache, bin, logs) before the backend is imported
_HOME = Path(tempfile.mkdtemp(prefix="autotuner-test-"))
os.environ["AUTOTUNER_HOME"] = str(_HOME)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import hardware, model_repository  # noqa: E402
from backend.hardware import GPUInfo, HardwareProfile  # noqa: E402


def make_hw(vram=8.0, ram=32.0, avail=24.0, vendor="NVIDIA", integrated=False, cores=8) -> HardwareProfile:
    return HardwareProfile(
        os_name="Windows", os_version="11", cpu_name="Test CPU",
        physical_cores=cores, logical_cores=cores * 2, cpu_freq_ghz=3.5,
        total_ram_gb=ram, available_ram_gb=avail,
        gpu=GPUInfo(name=f"Test {vendor} GPU", vendor=vendor, total_vram_gb=vram,
                    free_vram_gb=vram * 0.95 if not integrated else vram,
                    cuda=vendor == "NVIDIA", vulkan=True, integrated=integrated),
        score=50.0,
    )


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No network and a deterministic machine for every test."""
    monkeypatch.setattr(model_repository, "_fetch_hf", lambda: model_repository._fallback())
    monkeypatch.setattr(model_repository, "_mem_cache", None)
    model_repository.CUSTOM_FILE.unlink(missing_ok=True)
    from backend import benchmark_provider, runtime
    monkeypatch.setattr(benchmark_provider, "_refresh_in_background", lambda: None)
    monkeypatch.setattr(runtime, "releases_nonblocking", lambda: None)
    model_repository.CACHE_FILE.unlink(missing_ok=True)
    hw = make_hw()
    monkeypatch.setattr(hardware, "profile_hardware", lambda: hw)
    for mod in ("backend.selector", "backend.app"):
        if mod in sys.modules:
            monkeypatch.setattr(sys.modules[mod], "profile_hardware", lambda: hw)
    yield


@pytest.fixture
def home() -> Path:
    return _HOME
