"""
OpenCode integration — use the local model as the coding agent's provider.

OpenCode (https://opencode.ai) reads ~/.config/opencode/opencode.json. We add one provider,
"llm-autotuner", that points at the running llama.cpp / MLX server through OpenAI-compatible
chat completions, and optionally make its model the default. Everything else in the user's
config is left as it was; the previous file is kept as opencode.json.bak.
"""
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import LLM_PORT, get_logger, llm_api_key, safe_model_path

logger = get_logger("opencode")

PROVIDER_ID = "llm-autotuner"
SCHEMA = "https://opencode.ai/config.json"
# OpenCode's system prompt and tool definitions take ~10k tokens before the first message
MIN_CONTEXT = 32768

INSTALL = {
    "windows": ["npm install -g opencode-ai", "scoop install opencode"],
    "macos": ["curl -fsSL https://opencode.ai/install | bash", "brew install sst/tap/opencode"],
    "linux": ["curl -fsSL https://opencode.ai/install | bash", "npm install -g opencode-ai"],
}


def _platform() -> str:
    return "windows" if sys.platform == "win32" else "macos" if sys.platform == "darwin" else "linux"


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "opencode"


def config_path() -> Path:
    """The file OpenCode reads: opencode.json, or an existing opencode.jsonc."""
    d = config_dir()
    if not (d / "opencode.json").exists() and (d / "opencode.jsonc").exists():
        return d / "opencode.jsonc"
    return d / "opencode.json"


def find_exe() -> Optional[str]:
    exe = shutil.which("opencode")
    if exe:
        return exe
    home = Path.home()
    names = ["opencode.cmd", "opencode.exe"] if sys.platform == "win32" else ["opencode"]
    for d in (home / ".opencode" / "bin", home / ".local" / "bin", home / "bin",
              Path(os.environ.get("APPDATA", home)) / "npm", Path("/opt/homebrew/bin"), Path("/usr/local/bin")):
        for n in names:
            if (d / n).is_file():
                return str(d / n)
    return None


def _version(exe: str) -> str:
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=5,
                             creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        m = re.search(r"\d+\.\d+\.\d+", out.stdout or "")
        return m.group(0) if m else ""
    except Exception:
        return ""


# ─── Config file ───────────────────────────────────────────────────────────────

def strip_jsonc(text: str) -> str:
    """Remove // and /* */ comments and trailing commas, leaving strings untouched."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            while i < n and text[i] not in "\r\n":
                i += 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
        else:
            if c == ",":  # drop a trailing comma before } or ]
                j = i + 1
                while j < n and (text[j].isspace() or text.startswith("//", j) or text.startswith("/*", j)):
                    if text.startswith("//", j):
                        j = text.find("\n", j) if "\n" in text[j:] else n
                    elif text.startswith("/*", j):
                        e = text.find("*/", j + 2)
                        j = n if e < 0 else e + 2
                    else:
                        j += 1
                if j < n and text[j] in "}]":
                    i += 1
                    continue
            out.append(c)
            i += 1
    return "".join(out)


def read_config() -> Dict[str, Any]:
    p = config_path()
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        data = json.loads(strip_jsonc(text))
    except ValueError as e:
        raise ValueError(f"{p} is not valid JSON ({e}). Fix or remove it, then try again.")
    if not isinstance(data, dict):
        raise ValueError(f"{p} does not contain a JSON object.")
    return data


def _write_config(data: Dict[str, Any]) -> Path:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        shutil.copy2(p, p.with_name(p.name + ".bak"))
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def model_key(filename: str) -> str:
    """The model name the server answers to: llama-server's --alias is the file name."""
    return Path(filename.replace("\\", "/")).name


def provider_block(filename: str, ctx: int, display: str = "") -> Dict[str, Any]:
    key = model_key(filename)
    ctx = max(2048, int(ctx))
    options: Dict[str, Any] = {"baseURL": f"http://127.0.0.1:{LLM_PORT}/v1"}
    api_key = llm_api_key()
    if api_key:
        options["apiKey"] = api_key
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "LLM Autotuner (local)",
        "options": options,
        "models": {key: {"name": (display or key) + " (local)", "tool_call": True,
                         "limit": {"context": ctx, "output": min(16384, max(1024, ctx // 4))}}},
    }


def configure(filename: str, ctx: int, display: str = "", set_default: bool = True) -> Dict[str, Any]:
    safe_model_path(filename)  # same path rules as launching
    data = read_config()
    data.setdefault("$schema", SCHEMA)
    providers = data.get("provider")
    if not isinstance(providers, dict):
        providers = {}
    # One model at a time is served, so the provider lists only the current one
    providers[PROVIDER_ID] = provider_block(filename, ctx, display)
    data["provider"] = providers
    if set_default:
        data["model"] = f"{PROVIDER_ID}/{model_key(filename)}"
    p = _write_config(data)
    logger.info(f"OpenCode configured for {model_key(filename)} in {p}")
    return status()


def sync_api_key() -> bool:
    """After the API key changes, update it in OpenCode's config if we configured it."""
    try:
        data = read_config()
        prov = (data.get("provider") or {}).get(PROVIDER_ID)
        if not isinstance(prov, dict):
            return False
        opts = prov.setdefault("options", {})
        key = llm_api_key()
        if key:
            opts["apiKey"] = key
        else:
            opts.pop("apiKey", None)
        _write_config(data)
        return True
    except Exception as e:
        logger.warning(f"Could not update the OpenCode API key: {e}")
        return False


def status() -> Dict[str, Any]:
    exe = find_exe()
    out: Dict[str, Any] = {
        "installed": bool(exe), "path": exe or "", "version": _version(exe) if exe else "",
        "config_path": str(config_path()), "configured": False, "model": "", "is_default": False,
        "error": "", "install": INSTALL[_platform()], "min_context": MIN_CONTEXT,
        "provider_id": PROVIDER_ID,
    }
    try:
        data = read_config()
        prov = (data.get("provider") or {}).get(PROVIDER_ID)
        if isinstance(prov, dict):
            out["configured"] = True
            out["model"] = next(iter(prov.get("models") or {}), "")
            out["is_default"] = data.get("model") == f"{PROVIDER_ID}/{out['model']}"
    except ValueError as e:
        out["error"] = str(e)
    return out


# ─── Open a terminal running OpenCode ──────────────────────────────────────────

_LINUX_TERMINALS = [
    ("x-terminal-emulator", ["-e"]), ("gnome-terminal", ["--"]), ("konsole", ["-e"]),
    ("xfce4-terminal", ["-x"]), ("kitty", []), ("alacritty", ["-e"]), ("wezterm", ["start", "--"]),
    ("foot", []), ("xterm", ["-e"]),
]


def open_terminal(folder: str = "") -> Dict[str, Any]:
    exe = find_exe()
    if not exe:
        raise FileNotFoundError("OpenCode is not installed. Install it with one of the commands shown, "
                                "then try again.")
    cwd = Path(folder).expanduser() if folder.strip() else Path.home()
    if not cwd.is_dir():
        raise ValueError(f"Folder not found: {cwd}")
    cwd = cwd.resolve()
    plat = _platform()
    if plat == "windows":
        subprocess.Popen(["cmd.exe", "/k", exe], cwd=str(cwd),
                         creationflags=subprocess.CREATE_NEW_CONSOLE)  # type: ignore[attr-defined]
    elif plat == "macos":
        line = f"cd {shlex.quote(str(cwd))} && {shlex.quote(exe)}"
        script = line.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.Popen(["osascript", "-e", 'tell application "Terminal" to activate',
                          "-e", f'tell application "Terminal" to do script "{script}"'])
    else:
        for term, flag in _LINUX_TERMINALS:
            path = shutil.which(term)
            if path:
                subprocess.Popen([path, *flag, exe], cwd=str(cwd), start_new_session=True)
                break
        else:
            raise FileNotFoundError("No terminal app found. Open a terminal and run: opencode")
    return {"ok": True, "folder": str(cwd)}
