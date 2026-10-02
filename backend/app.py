import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .hardware import HardwareProfile, profile_hardware
import httpx

from .model_repository import get_models, quant_files, REPO_ID_RE, add_custom_model, remove_custom_model
from .selector import build_table, get_recommendation
from .downloader import (resolve_download, start_download, all_jobs, installed_models, cancel_job,
                         delete_model, disk_info, DiskSpaceError)
from .launcher import launch, stop, status as launch_status
from . import runtime
from .benchmark_provider import get_arena, _refresh_in_background as refresh_arena
from .utils import (APP_NAME, APP_VERSION, FRONTEND_DIR, LLM_PORT, get_logger, get_setting, set_setting,
                    hf_token, llm_api_key, regenerate_api_key)

logger = get_logger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # Never leave an orphaned llama-server holding GPU memory after the app quits
    stop()


app = FastAPI(
    title="LLM Autotuner",
    description="Hardware-aware local LLM recommendation and deployment engine.",
    version=APP_VERSION,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# The API controls local processes and the filesystem, so it must only be reachable
# from this machine: reject foreign Host headers (DNS rebinding) and cross-site requests.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.middleware("http")
async def same_origin_only(request: Request, call_next):
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin:
            host = request.headers.get("host", "")
            if origin not in (f"http://{host}", f"https://{host}"):
                return JSONResponse({"detail": "Cross-origin request blocked"}, status_code=403)
    return await call_next(request)


def _hw(override: Optional[HardwareProfile] = None) -> HardwareProfile:
    return override or profile_hardware()


# ─── Meta ──────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def api_health():
    return {"app": APP_NAME, "version": APP_VERSION, "ok": True}


# ─── Hardware ──────────────────────────────────────────────────────────────────

@app.get("/api/hardware")
def api_hardware():
    return profile_hardware()


# ─── Model Table ───────────────────────────────────────────────────────────────

@app.get("/api/models")
def api_models():
    return build_table()


@app.post("/api/models/override")
def api_models_override(req: HardwareProfile):
    return build_table(req)


@app.post("/api/refresh")
def api_refresh():
    get_models(force=True)
    refresh_arena()
    return {"ok": True, "count": len(build_table())}


class CustomModelReq(BaseModel):
    url: str = Field(..., max_length=500)


@app.post("/api/models/custom")
def api_add_custom(req: CustomModelReq):
    try:
        e = add_custom_model(req.url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except (FileNotFoundError, LookupError) as e:
        raise HTTPException(404, str(e))
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Could not reach HuggingFace: {e}")
    return {"ok": True, "id": e.id, "name": e.name}


@app.delete("/api/models/custom/{model_id:path}")
def api_remove_custom(model_id: str):
    if not remove_custom_model(model_id):
        raise HTTPException(404, "Not a custom model")
    return {"ok": True}


@app.get("/api/benchmarks/info")
def api_bench_info():
    a = get_arena(refresh=False)
    return {"source": "LMArena (lmarena-ai/leaderboard-dataset)", "published": a.get("published", ""),
            "models": len(a.get("text") or {}), "fetched": a.get("ts", 0)}


# ─── Recommendation ────────────────────────────────────────────────────────────

class RecommendReq(BaseModel):
    model_id: str
    persona: str = "general"
    hardware: Optional[HardwareProfile] = None


@app.post("/api/recommend")
def api_recommend(req: RecommendReq):
    try:
        return get_recommendation(req.model_id, req.persona, _hw(req.hardware))
    except ValueError as e:
        raise HTTPException(404, str(e))


# ─── Downloads ─────────────────────────────────────────────────────────────────

class DownloadReq(BaseModel):
    model_id: str = Field(..., max_length=200)
    quant: str = Field(..., max_length=40)


@app.get("/api/hf-files/{model_id:path}")
def api_hf_files(model_id: str):
    """Real GGUF files of a HuggingFace repo, grouped by quantization, with exact sizes."""
    if not REPO_ID_RE.match(model_id):
        raise HTTPException(400, "Invalid model id")
    try:
        return quant_files(model_id)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.warning(f"hf-files {model_id}: {e}")
        raise HTTPException(502, f"Could not reach HuggingFace: {e}")


@app.post("/api/download")
async def api_download(req: DownloadReq):
    try:
        plan = await asyncio.to_thread(resolve_download, req.model_id, req.quant)
    except (ValueError, LookupError) as e:
        raise HTTPException(400, str(e))
    except DiskSpaceError as e:
        raise HTTPException(507, str(e))
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, f"Could not reach HuggingFace: {e}")
    return start_download(req.model_id, req.quant, plan)


@app.get("/api/disk")
def api_disk():
    return disk_info()


@app.get("/api/downloads")
def api_downloads():
    return all_jobs()


@app.delete("/api/downloads/{key:path}")
def api_cancel(key: str):
    if not cancel_job(key):
        raise HTTPException(404, "No such download")
    return {"ok": True}


@app.get("/api/installed")
def api_installed():
    return installed_models()


@app.delete("/api/installed/{filename:path}")
def api_delete(filename: str):
    st = launch_status()
    if st.running and st.model == filename:
        stop()
    try:
        ok = delete_model(filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not ok:
        raise HTTPException(404, "File not found")
    return {"ok": True}


# ─── llama.cpp runtime ─────────────────────────────────────────────────────────

@app.get("/api/runtime")
def api_runtime():
    g = profile_hardware().gpu
    return runtime.status(g.vendor, g.integrated, g.driver_version)


class RuntimeInstallReq(BaseModel):
    variant: Optional[str] = Field(None, max_length=60, pattern=r"^[a-z0-9.\-]+$")


@app.post("/api/runtime/install")
async def api_runtime_install(req: Optional[RuntimeInstallReq] = None):
    """Install, update, or switch the llama.cpp build (variant e.g. 'win-vulkan-x64')."""
    g = (await asyncio.to_thread(profile_hardware)).gpu
    if not runtime.status(g.vendor, check_updates=False).installing:
        await asyncio.to_thread(stop)  # a running llama-server locks its files on Windows
    return runtime.start_install(g.vendor, g.integrated, g.driver_version, req.variant if req else None)


# ─── Settings ──────────────────────────────────────────────────────────────────

def _settings():
    tok = hf_token()
    key = llm_api_key()
    return {
        "hf_token_set": bool(tok),
        "hf_token_hint": (tok[:5] + "…" + tok[-3:]) if len(tok) > 10 else "",
        "hf_user": get_setting("hf_user", ""),
        "api_key": key,
        "api_key_enabled": bool(key),
        "llm_base_url": f"http://127.0.0.1:{LLM_PORT}/v1",
        "disk": disk_info(),
    }


@app.get("/api/settings")
def api_settings():
    return _settings()


class TokenReq(BaseModel):
    token: str = Field(..., min_length=8, max_length=200)


@app.post("/api/settings/hf-token")
def api_set_token(req: TokenReq):
    tok = req.token.strip()
    try:
        r = httpx.get("https://huggingface.co/api/whoami-v2", timeout=15,
                      headers={"Authorization": f"Bearer {tok}"})
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Could not reach HuggingFace to check the token: {e}")
    if r.status_code != 200:
        raise HTTPException(400, "HuggingFace rejected this token. Create a 'Read' token at "
                                 "huggingface.co/settings/tokens and paste it here.")
    set_setting("hf_token", tok)
    set_setting("hf_user", r.json().get("name", ""))
    return _settings()


@app.delete("/api/settings/hf-token")
def api_clear_token():
    set_setting("hf_token", None)
    set_setting("hf_user", None)
    return _settings()


@app.post("/api/settings/api-key/regenerate")
def api_regen_key():
    regenerate_api_key()
    return _settings()


# ─── Launcher ──────────────────────────────────────────────────────────────────

class LaunchReq(BaseModel):
    filename: str = Field(..., max_length=400)
    ctx: int = 4096
    threads: int = 4
    gpu_layers: int = 0


@app.post("/api/launch")
def api_launch(req: LaunchReq):
    return launch(req.filename, req.ctx, req.threads, req.gpu_layers)


@app.post("/api/stop")
def api_stop():
    return {"stopped": stop()}


@app.get("/api/launch-status")
def api_launch_status():
    return launch_status()


# ─── Frontend ──────────────────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"), headers={"Cache-Control": "no-cache"})
