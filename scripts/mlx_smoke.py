"""
End-to-end MLX check against a running app (used by CI on an Apple Silicon runner):
install the MLX runtime → download a small MLX model → launch → chat through the gateway.

    python main.py --no-browser --port 8765 &
    python scripts/mlx_smoke.py http://127.0.0.1:8765
"""
import sys
import time

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765").rstrip("/")
MODEL = "mlx-community/SmolLM2-135M-Instruct-8bit"


def wait(what, fn, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            print(f"✓ {what} ({time.time() - t0:.0f}s)")
            return v
        time.sleep(2)
    sys.exit(f"✗ timed out waiting for {what}")


c = httpx.Client(base_url=BASE, timeout=60)
wait("app up", lambda: c.get("/api/health").status_code == 200, 60)

st = c.get("/api/runtime/mlx").json()
assert st["supported"], f"MLX not supported here: {st}"
c.post("/api/runtime/mlx/install", json={"upgrade": False})


def mlx_ready():
    s = c.get("/api/runtime/mlx").json()
    if s["stage"] == "failed":
        sys.exit(f"✗ MLX install failed: {s['error']}")
    return s["installed"] and not s["installing"] and s
print("mlx-lm", wait("MLX runtime installed", mlx_ready, 900)["version"])

job = c.post("/api/download", json={"model_id": MODEL, "quant": "8bit"}).json()
print("download job:", job.get("key"), job.get("filename"))


def downloaded():
    j = next((x for x in c.get("/api/downloads").json() if x["key"] == job["key"]), {})
    if j.get("status") == "failed":
        sys.exit(f"✗ download failed: {j.get('error')}")
    return j.get("status") == "done"
wait("model downloaded", downloaded, 600)

r = c.post("/api/launch", json={"filename": job["filename"], "ctx": 2048, "threads": 4, "gpu_layers": 0}).json()
assert r["running"], f"launch failed: {r['message']}"
st = wait("model ready", lambda: (lambda s: s["ready"] and s)(c.get("/api/launch-status").json()), 300)
assert st["backend"] == "mlx", st

key = c.get("/api/settings").json()["api_key"]
body = {"model": "anything", "max_tokens": 20, "messages": [{"role": "user", "content": "Say hello."}]}
no_key = httpx.post(st["base_url"] + "/chat/completions", json=body, timeout=120)
assert no_key.status_code == 401, f"expected 401 without key, got {no_key.status_code}"
resp = httpx.post(st["base_url"] + "/chat/completions", json=body, timeout=300,
                  headers={"Authorization": f"Bearer {key}"})
print("chat:", resp.status_code, resp.text[:300])
assert resp.status_code == 200, resp.text
text = resp.json()["choices"][0]["message"]["content"]
assert text.strip(), "empty completion"
print("✓ MLX chat works:", repr(text[:80]))
c.post("/api/stop")
