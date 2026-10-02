## Downloads

| Your computer | File | How to start |
|---|---|---|
| **Windows** 10/11 (64-bit) | `AI_Model_Autotuner-windows-x64.exe` | Put it in its own folder and double-click it. If SmartScreen warns, click **More info → Run anyway**. |
| **Mac** with Apple Silicon (M1–M4) | `AI_Model_Autotuner-macos-arm64.zip` | Unzip, then double-click **Start LLM Autotuner.command**. If macOS blocks it: System Settings → Privacy & Security → **Open Anyway** (details in the included README.txt). |
| **Linux** (x64) | `AI_Model_Autotuner-linux-x64.tar.gz` | `tar xzf AI_Model_Autotuner-linux-x64.tar.gz && ./llm-autotuner/AI_Model_Autotuner` |

Intel Macs and other systems: see **Run from source** in the README.

## What's new

- **MLX models on Apple Silicon Macs**: about 200 models from mlx-community (marked **MLX**), run with Apple's own MLX engine, which is often faster than GGUF on M-series chips. The app sets up the MLX engine by itself the first time you launch an MLX model (about 250 MB, one time); no Python or Homebrew needed.
- MLX models use the same address and API key as GGUF models, so connected apps keep working.
- **＋ Add model** also accepts links to MLX repositories.
- Pro mode has a new **Format** filter (GGUF / MLX) on Macs.

### Earlier in 2.2

- Mac and Linux downloads; CUDA build for NVIDIA GPUs; API-key protected model server; disk-space checks; catalog from four publishers with a source picker; HuggingFace token for gated models; live LMArena quality data.
