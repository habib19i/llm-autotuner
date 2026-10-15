## Downloads

| Your computer | File | How to start |
|---|---|---|
| **Windows** 10/11 (64-bit) | `AI_Model_Autotuner-windows-x64.exe` | Put it in its own folder and double-click it. If SmartScreen warns, click **More info → Run anyway**. |
| **Mac** with Apple Silicon (M1–M4) | `AI_Model_Autotuner-macos-arm64.zip` | Unzip, then double-click **Start LLM Autotuner.command**. If macOS blocks it: System Settings → Privacy & Security → **Open Anyway** (details in the included README.txt). |
| **Linux** (x64) | `AI_Model_Autotuner-linux-x64.tar.gz` | `tar xzf AI_Model_Autotuner-linux-x64.tar.gz && ./llm-autotuner/AI_Model_Autotuner` |

Intel Macs and other systems: see **Run from source** in the README.

## What's new

- **Deploy with OpenCode**: the Deploy page can add the model to [OpenCode](https://opencode.ai)'s settings in one click (your other OpenCode settings are kept; the old file is saved as `.bak`), make it the default model, and open OpenCode on a project folder in a new terminal. Models that fit with a 32k or 64k context offer to launch with it, since coding agents need the room. Regenerating the API key updates OpenCode's settings too.
- **More models**: the catalog now also reads the GGUF releases of Qwen, Google, IBM Granite, Microsoft and Nous Research, plus the mradermacher and MaziyarPanahi collections, and scans more repositories from every publisher. **Find more models** (Pro mode) searches all of HuggingFace and adds any result with one click. Pro mode has a full provider list and a **Size** filter.
- **New design**: one memory bar runs through the app (your memory on the first screen, each model's footprint in the lists), a bundled typeface (no web fonts), a light and a dark theme that can follow the system (Settings → Appearance), and a workflow step that previews its top pick as you point at each option.
- llama.cpp now starts with `--jinja`, so tool calls work for agents.

### Earlier in 2.3

- Pro mode now has a **Deploy →** button that opens the Deploy page (connection details, API key, launch) for the selected model, downloading it first if needed.
- **MLX models on Apple Silicon Macs**: about 200 models from mlx-community (marked **MLX**), run with Apple's own MLX engine, which is often faster than GGUF on M-series chips. The app sets up the MLX engine by itself the first time you launch an MLX model (about 250 MB, one time); no Python or Homebrew needed.
- MLX models use the same address and API key as GGUF models, so connected apps keep working.
- **＋ Add model** also accepts links to MLX repositories.
- Pro mode has a new **Format** filter (GGUF / MLX) on Macs.

### Earlier in 2.2

- Mac and Linux downloads; CUDA build for NVIDIA GPUs; API-key protected model server; disk-space checks; catalog from four publishers with a source picker; HuggingFace token for gated models; live LMArena quality data.
