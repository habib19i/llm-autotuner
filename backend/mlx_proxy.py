"""
Authenticating gateway in front of mlx_lm.server.

mlx_lm.server has no API key and answers any web origin, so it runs on a private port and
this gateway listens on the public model port instead. It:
  - requires the app's API key (Authorization: Bearer … or x-api-key), like llama-server;
  - rewrites the request's "model" field to the loaded model — otherwise mlx_lm.server
    treats the name a client sends as a HuggingFace repo and tries to download it;
  - streams responses through unchanged (server-sent events included).
"""
import json
import threading
from typing import Optional

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .utils import get_logger

logger = get_logger("mlx_proxy")

_HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
        "proxy-authorization", "proxy-authenticate", "host", "content-length"}
_REWRITE_PATHS = ("/v1/chat/completions", "/v1/completions", "/chat/completions", "/completions")


def build_app(upstream: str, api_key: str, transport: Optional[httpx.AsyncBaseTransport] = None) -> Starlette:
    client = httpx.AsyncClient(base_url=upstream, timeout=httpx.Timeout(10.0, read=None),
                               transport=transport)

    async def handle(request: Request):
        path = request.url.path
        if request.method != "OPTIONS" and path != "/health" and api_key:
            auth = request.headers.get("authorization", "")
            token = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
            if token != api_key:
                return JSONResponse({"error": {"message": "Invalid API key", "type": "authentication_error",
                                               "code": 401}}, status_code=401)
        body = await request.body()
        if request.method == "POST" and path in _REWRITE_PATHS and body:
            try:
                data = json.loads(body)
                if isinstance(data, dict):
                    data["model"] = "default_model"
                    body = json.dumps(data).encode()
            except ValueError:
                pass
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in _HOP and k.lower() not in ("authorization", "x-api-key")}
        req = client.build_request(request.method, path, params=request.query_params, content=body,
                                   headers=headers)
        try:
            resp = await client.send(req, stream=True)
        except httpx.HTTPError as e:
            return JSONResponse({"error": {"message": f"Model server not ready: {e}", "code": 503}},
                                status_code=503)
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP}

        async def body_iter():
            try:
                if resp.is_stream_consumed:  # body already buffered by the transport
                    yield resp.content
                    return
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(body_iter(), status_code=resp.status_code, headers=out_headers)

    methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
    app = Starlette(routes=[Route("/{path:path}", handle, methods=methods)])
    app.state.client = client
    return app


class Gateway:
    """Runs the gateway with uvicorn in a background thread."""

    def __init__(self, port: int, upstream: str, api_key: str):
        self.port = port
        config = uvicorn.Config(build_app(upstream, api_key), host="127.0.0.1", port=port,
                                log_level="warning", lifespan="off")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, name="mlx-gateway", daemon=True)

    def start(self, timeout: float = 5.0) -> bool:
        self.thread.start()
        import time
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.server.started:
                return True
            if not self.thread.is_alive():
                return False
            time.sleep(0.05)
        return self.server.started

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)
