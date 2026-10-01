import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .hardware import HardwareProfile, profile_hardware
from .model_repository import get_models, quant_files, REPO_ID_RE
from .selector import build_table, get_recommendation
from .downloader import resolve_download, start_download, all_jobs, installed_models, cancel_job, delete_model
from .launcher import launch, stop, status as launch_status
from . import runtime
from .utils import APP_NAME, APP_VERSION, FRONTEND_DIR, get_logger

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
    return {"ok": True, "count": len(build_table())}


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
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, f"Could not reach HuggingFace: {e}")
    return start_download(req.model_id, req.quant, plan)


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
    return runtime.status(g.vendor, g.integrated)


@app.post("/api/runtime/install")
async def api_runtime_install():
    g = (await asyncio.to_thread(profile_hardware)).gpu
    return runtime.start_install(g.vendor, g.integrated)


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
