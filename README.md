# LLM Autotuner

Hardware-aware local LLM deployment. It scans your PC, ranks ~160 current GGUF models
from HuggingFace (Unsloth) by how well they fit **your** memory, downloads the right
quantization, installs the llama.cpp runtime, and starts an OpenAI-compatible API server —
all from a local web UI.

- **Simple mode** — a 5-step wizard: scan → hardware → workflow → model → deploy.
- **Pro mode** — the full sortable/filterable model table, per-workflow tuning, exact
  per-quant file sizes, download / launch / delete.

Everything runs on `127.0.0.1`; nothing is exposed to the network.

## Using the app (end users)

1. Download `AI_Model_Autotuner.exe` and put it in its own folder (it stores data next to itself).
2. Double-click it. Your browser opens at `http://127.0.0.1:8001/`.
3. Follow the wizard. On first launch the app downloads the llama.cpp runtime automatically
   (one time).
4. When the model is running, connect any OpenAI-compatible client:

   | Setting  | Value                          |
   |----------|--------------------------------|
   | Base URL | `http://127.0.0.1:8080/v1`     |
   | API key  | any value (e.g. `sk-local`)    |
   | Model    | the `.gguf` file name shown in the app |

   llama.cpp's own chat UI is at `http://127.0.0.1:8080` ("Open chat ↗" in the app).

Closing the console window stops the app and the model server.

### Data folders (created next to the .exe)

| Folder    | Contents |
|-----------|----------|
| `models/` | Downloaded models, one folder per repo (multi-part models and vision projectors are kept together) |
| `bin/`    | The llama.cpp runtime |
| `cache/`  | Model catalog cache (refreshed every 24 h, or via **Refresh** in Pro mode) |
| `logs/`   | `llama-server.log` — check this if a model fails to start |

### Options

```
AI_Model_Autotuner.exe [--port 8001] [--no-browser]
```

| Environment variable  | Default | Purpose |
|-----------------------|---------|---------|
| `AUTOTUNER_PORT`      | `8001`  | Web UI port (falls back to the next free port if busy) |
| `AUTOTUNER_LLM_PORT`  | `8080`  | Port of the model API server |
| `AUTOTUNER_HOME`      | exe folder | Where `models/`, `bin/`, `cache/`, `logs/` live |

If you already have `llama-server` on your `PATH` or in `bin/`, it is used instead of
downloading one.

## Development

Requires Python 3.10+.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe main.py --browser      # http://127.0.0.1:8001
.venv\Scripts\python.exe -m pytest              # offline test suite
```

API docs are served at `http://127.0.0.1:8001/api/docs`.

### Project layout

```
main.py                    entry point (port selection, single-instance check, browser)
backend/
  app.py                   FastAPI routes + localhost-only guards
  hardware.py              CPU / RAM / GPU detection (NVIDIA, AMD, Intel, Apple; iGPU aware)
  model_repository.py      HuggingFace catalog (GGUF metadata), quant parsing, offline fallback
  scoring.py               memory fit, GPU offload, context size, tok/s estimate
  selector.py              table rows + per-workflow recommendations
  benchmark_provider.py    published benchmarks, size-based estimates otherwise (marked ~)
  downloader.py            resumable multi-file downloads, installed-model registry
  runtime.py               llama.cpp release download/installation
  launcher.py              llama-server process management + health check
frontend/index.html        single-file UI (EN / 中文, dark / light)
tests/                     pytest suite (no network needed)
```

### Building the Windows executable

Build from a clean virtual environment (a full Anaconda base env makes PyInstaller fail —
see the notes at the top of `build.py`):

```powershell
python -m venv .buildvenv
.buildvenv\Scripts\python.exe -m pip install -r requirements-dev.txt
.buildvenv\Scripts\python.exe build.py          # runs the tests, then PyInstaller
```

The result is `dist\AI_Model_Autotuner.exe` (~20 MB, single file). macOS / Linux builds
work the same way with `python build.py` on those platforms.

### Release checklist

1. `python -m pytest` passes.
2. `python build.py` produces `dist/AI_Model_Autotuner.exe`.
3. Copy the exe to an empty folder, run it, and complete the wizard with a small model
   (e.g. *SmolLM2 135M* in Pro mode) to confirm download → runtime install → launch → chat.
4. Publish the exe (e.g. as a GitHub release asset). Windows SmartScreen will warn about an
   unsigned executable; code-sign it to avoid that.

## How recommendations work

- **Size**: parameter counts and context lengths come from the GGUF metadata on HuggingFace;
  file sizes in the download picker are the exact files.
- **Fit**: weights + KV cache (at the context the app will launch with: 16k, falling back to
  8k/4k) + buffers, against free VRAM, or against free RAM for integrated GPUs and Apple
  Silicon (unified memory is never counted twice). Models that only fit below ~3 bits per
  weight are flagged *Marginal*.
- **Speed**: token generation is memory-bandwidth bound, so tok/s ≈ bandwidth ÷ bytes of
  *active* weights — which is why mixture-of-experts models (e.g. `30B-A3B`) run fast.
- **Benchmarks**: published scores for known models; others are estimated from size and
  shown with `~`.

## License

MIT — see [LICENSE](LICENSE).
