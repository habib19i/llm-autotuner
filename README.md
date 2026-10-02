# LLM Autotuner

Hardware-aware local LLM deployment. It scans your computer, ranks ~290 current GGUF models
(plus ~200 MLX models on Apple Silicon Macs) from HuggingFace by how well they fit **your**
memory, downloads the right quantization,
installs the llama.cpp runtime, and starts an OpenAI-compatible API server — all from a
local web UI. Windows, macOS and Linux.

- **Simple mode** — a 5-step wizard: scan → hardware → workflow → model → deploy.
- **Pro mode** — the full sortable/filterable model table, per-workflow tuning, exact
  per-quant file sizes, multiple download sources, download / launch / delete.

Everything runs on `127.0.0.1`; nothing is exposed to the network.

## Download

Get the latest version from the **[Releases page](https://github.com/habib19i/llm-autotuner/releases/latest)**:

| Your computer | File | How to start |
|---|---|---|
| Windows 10/11 (64-bit) | `AI_Model_Autotuner-windows-x64.exe` | Put it in its own folder and double-click it. If SmartScreen warns, click **More info → Run anyway**. |
| Mac with Apple Silicon (M1–M4) | `AI_Model_Autotuner-macos-arm64.zip` | Unzip, double-click **Start LLM Autotuner.command**. If macOS blocks it: **System Settings → Privacy & Security → Open Anyway**. |
| Linux (x64) | `AI_Model_Autotuner-linux-x64.tar.gz` | `tar xzf AI_Model_Autotuner-linux-x64.tar.gz && ./llm-autotuner/AI_Model_Autotuner` |

Intel Macs and other systems: [run from source](#run-from-source).

## Using the app

1. Start it as above. Your browser opens at `http://127.0.0.1:8001/`.
2. Follow the wizard. On first launch the app downloads the right llama.cpp build for your
   machine automatically (CUDA for NVIDIA GPUs, Vulkan for AMD/Intel, Metal on Mac).
3. When the model is running, connect any OpenAI-compatible client (VS Code, Cursor,
   Open WebUI, Obsidian…):

   | Setting  | Value |
   |----------|-------|
   | Base URL | `http://127.0.0.1:8080/v1` |
   | API key  | shown in the app (⚙ Settings, and on the Deploy step) |
   | Model    | the `.gguf` file name shown in the app |

   Or just chat inside the app: click **💬 Chat** in the bottom bar (or on the Deploy page).
   Replies stream live, reasoning models show their thinking in a collapsible section, vision
   models accept images (📎), and each answer shows its speed in tokens per second.

Closing the app's window stops the app and the model server.

### Settings (⚙)

- **HuggingFace token** — only needed for gated models (e.g. Llama). Accept the model's
  license on HuggingFace, create a *Read* token at huggingface.co/settings/tokens, paste it.
- **Model API key** — required by the running model so other websites open in your browser
  can't use it. Copy it or generate a new one.
- **llama.cpp runtime** — current version and build, one-click updates, and a switch to the
  Vulkan build if the CUDA build doesn't start on your PC.
- **Storage** — where models are stored and how much space is free. Every download is
  checked against free disk space first.

### More models

- The catalog combines **Unsloth, bartowski, LM Studio Community and ggml-org**. When several
  publish the same model, the download dialog lets you pick the source.
- **＋ Add model** (Pro mode) accepts any HuggingFace link to a repository with `.gguf` files
  or MLX weights.

### MLX models (Apple Silicon Macs)

On a Mac with an M-series chip the catalog also lists ~200 **MLX** models from
[mlx-community](https://huggingface.co/mlx-community) (marked **MLX**; use the *Format* filter in
Pro mode). MLX is Apple's own machine-learning engine and is often faster than GGUF on Apple
Silicon.

- The first time you launch an MLX model the app sets up the MLX engine by itself (about
  250 MB, one time) in its own `bin/` folder — no Python or Homebrew needed. Update it from
  ⚙ Settings.
- MLX models are served on the same address (`http://127.0.0.1:8080/v1`) with the same API
  key, so apps connected to the GGUF version keep working.
- Already have `mlx-lm` installed? Set `AUTOTUNER_MLX_PYTHON` to that Python to use it.

### Data folders (created next to the app)

| Folder    | Contents |
|-----------|----------|
| `models/` | Downloaded models, one folder per repo (multi-part models and vision projectors are kept together) |
| `bin/`    | The llama.cpp runtime |
| `cache/`  | Model catalog, quality ratings and settings |
| `logs/`   | `model-server.log` — check this if a model fails to start |

### Options

```
AI_Model_Autotuner [--port 8001] [--no-browser]
```

| Environment variable  | Default | Purpose |
|-----------------------|---------|---------|
| `AUTOTUNER_PORT`      | `8001`  | Web UI port (falls back to the next free port if busy) |
| `AUTOTUNER_LLM_PORT`  | `8080`  | Port of the model API server |
| `AUTOTUNER_HOME`      | app folder | Where `models/`, `bin/`, `cache/`, `logs/` live |
| `AUTOTUNER_API_KEY`   | generated | Fixed API key for the model server (empty string disables it) |
| `HF_TOKEN`            | —       | HuggingFace token (alternative to Settings) |
| `AUTOTUNER_MLX_PYTHON`| —       | Python with `mlx-lm` to use instead of the app-managed MLX engine |
| `AUTOTUNER_ENABLE_MLX`| —       | `1` shows MLX models on non-Apple machines (development only) |

If you already have `llama-server` on your `PATH` or in `bin/`, it is used instead of
downloading one.

## Run from source

Works on Windows, macOS (Intel and Apple Silicon) and Linux. Requires Python 3.10+
(macOS: install from [python.org](https://www.python.org/downloads/macos/)).

**macOS / Linux**

```bash
git clone https://github.com/habib19i/llm-autotuner.git
cd llm-autotuner
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --browser
```

Next time: `cd llm-autotuner && source .venv/bin/activate && python main.py --browser`.

**Windows**

```powershell
git clone https://github.com/habib19i/llm-autotuner.git
cd llm-autotuner
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py --browser
```

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest              # offline test suite
```

API docs are served at `http://127.0.0.1:8001/api/docs`. CI runs the tests on Windows,
macOS and Linux for every push.

### Project layout

```
main.py                    entry point (port selection, single-instance check, browser)
backend/
  app.py                   FastAPI routes + localhost-only guards
  hardware.py              CPU / RAM / GPU detection (NVIDIA, AMD, Intel, Apple; iGPU aware)
  model_repository.py      HuggingFace catalog (4 publishers), custom models, quant parsing
  benchmark_provider.py    LMArena ratings, published benchmarks, calibrated estimates
  data/arena_snapshot.json bundled ratings for first run / offline
  scoring.py               memory fit, GPU offload, context size, tok/s estimate
  selector.py              table rows + per-workflow recommendations
  downloader.py            resumable multi-file downloads, disk checks, installed models
  runtime.py               llama.cpp build selection (CUDA/Vulkan/Metal/CPU), install, updates
  launcher.py              model server process management (llama.cpp or MLX), API key, health
  mlx_runtime.py           MLX engine setup on Apple Silicon (uv + private Python + mlx-lm)
  mlx_proxy.py             API-key gateway in front of mlx_lm.server
scripts/mlx_smoke.py       real MLX end-to-end check (runs in CI on an Apple Silicon Mac)
frontend/index.html        single-file UI (EN / 中文, dark / light)
packaging/                 release notes, macOS launcher, signing entitlements
.github/workflows/         CI tests + release builds for Windows, macOS, Linux
tests/                     pytest suite (no network needed)
```

### Building locally

Build from a clean virtual environment (a full Anaconda base env makes PyInstaller fail —
see the notes at the top of `build.py`):

```bash
python -m venv .buildvenv
# Windows: .buildvenv\Scripts\python.exe   macOS/Linux: .buildvenv/bin/python
<venv-python> -m pip install -r requirements-dev.txt
<venv-python> build.py          # runs the tests, then PyInstaller → dist/
```

### Releasing

Push a version tag; GitHub Actions builds Windows, macOS and Linux, smoke-tests each build,
and publishes a release with all three downloads and `packaging/RELEASE_NOTES.md`:

```bash
git tag v2.2.0
git push origin v2.2.0
```

### Code signing (optional)

Unsigned builds work, but Windows SmartScreen and macOS Gatekeeper show warnings. The release
workflow signs automatically when these repository secrets exist
(Settings → Secrets and variables → Actions):

| Platform | Secrets | Where to get a certificate |
|---|---|---|
| Windows | `WINDOWS_CERT_PFX` (base64 of the .pfx), `WINDOWS_CERT_PASSWORD` | A code-signing certificate from a CA (e.g. Sectigo, DigiCert), or [SignPath](https://signpath.org) (free for open source) |
| macOS | `MACOS_CERT_P12` (base64), `MACOS_CERT_PASSWORD`, `MACOS_CODESIGN_IDENTITY` (e.g. `Developer ID Application: Name (TEAMID)`), and for notarization `APPLE_ID`, `APPLE_TEAM_ID`, `APPLE_APP_PASSWORD` | [Apple Developer Program](https://developer.apple.com/programs/) |

## How recommendations work

- **Size**: parameter counts and context lengths come from the GGUF metadata on HuggingFace;
  file sizes in the download picker are the exact files.
- **Fit**: weights + KV cache (at the context the app will launch with: 16k, falling back to
  8k/4k) + buffers, against free VRAM, or against free RAM for integrated GPUs and Apple
  Silicon (unified memory is never counted twice). Models that only fit below ~3 bits per
  weight are flagged *Marginal*.
- **Speed**: token generation is memory-bandwidth bound, so tok/s ≈ bandwidth ÷ bytes of
  *active* weights — which is why mixture-of-experts models (e.g. `30B-A3B`) run fast.
- **Quality**: the *Score* column is based on [LMArena](https://lmarena.ai) ratings from
  its public leaderboard dataset (open-weight models, refreshed daily). Models that aren't on
  the leaderboard get a rating estimated from their size and release date, calibrated on the
  measured models, and are marked with `~`. Published benchmark results (MMLU, HumanEval…)
  are shown only for the exact models they were published for.

## License

MIT — see [LICENSE](LICENSE).
