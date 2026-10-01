import os
import sys
import logging
import subprocess
from pathlib import Path

APP_NAME = "llm-autotuner"
APP_VERSION = "2.1.0"

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running in a PyInstaller bundle
    # The bundled files (like frontend) are extracted to sys._MEIPASS
    BUNDLE_DIR = Path(sys._MEIPASS)
    # The user's models and cache should go alongside the executable
    RUNTIME_DIR = Path(sys.executable).parent
else:
    # Running in normal Python environment
    BUNDLE_DIR = Path(__file__).resolve().parent.parent
    RUNTIME_DIR = BUNDLE_DIR

# AUTOTUNER_HOME relocates all user data (models, cache, llama.cpp runtime, logs)
if os.environ.get("AUTOTUNER_HOME"):
    RUNTIME_DIR = Path(os.environ["AUTOTUNER_HOME"]).expanduser().resolve()

FRONTEND_DIR = BUNDLE_DIR / "frontend"
CACHE_DIR = RUNTIME_DIR / "cache"
MODELS_DIR = RUNTIME_DIR / "models"
BIN_DIR = RUNTIME_DIR / "bin"
LOG_DIR = RUNTIME_DIR / "logs"

for _d in (CACHE_DIR, MODELS_DIR, BIN_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

APP_HOST = "127.0.0.1"
APP_PORT = int(os.environ.get("AUTOTUNER_PORT", "8001"))
LLM_PORT = int(os.environ.get("AUTOTUNER_LLM_PORT", "8080"))

# Hide console windows for helper processes (PowerShell, llama-server) on Windows
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def format_bytes(n: int) -> str:
    if not n or n <= 0:
        return "0 B"
    gb = n / (1024 ** 3)
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    mb = n / (1024 ** 2)
    return f"{mb:.0f} MB"


def safe_model_path(rel: str) -> Path:
    """Resolve a user-supplied path relative to MODELS_DIR, refusing anything outside it."""
    if not rel or "\x00" in rel:
        raise ValueError("Invalid model path")
    base = MODELS_DIR.resolve()
    p = (base / rel).resolve()
    if p == base or base not in p.parents:
        raise ValueError("Path escapes the models directory")
    return p


def rel_model_path(p: Path) -> str:
    """Path relative to MODELS_DIR, always with forward slashes (stable API identifier)."""
    return p.resolve().relative_to(MODELS_DIR.resolve()).as_posix()
