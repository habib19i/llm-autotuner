## Downloads

| Your computer | File | How to start |
|---|---|---|
| **Windows** 10/11 (64-bit) | `AI_Model_Autotuner-windows-x64.exe` | Put it in its own folder and double-click it. If SmartScreen warns, click **More info → Run anyway**. |
| **Mac** with Apple Silicon (M1–M4) | `AI_Model_Autotuner-macos-arm64.zip` | Unzip, then double-click **Start LLM Autotuner.command**. If macOS blocks it: System Settings → Privacy & Security → **Open Anyway** (details in the included README.txt). |
| **Linux** (x64) | `AI_Model_Autotuner-linux-x64.tar.gz` | `tar xzf AI_Model_Autotuner-linux-x64.tar.gz && ./llm-autotuner/AI_Model_Autotuner` |

Intel Macs and other systems: see **Run from source** in the README.

## What's new

- **Mac and Linux downloads** alongside Windows.
- **Faster on NVIDIA GPUs**: installs the CUDA build of llama.cpp when your driver supports it (Vulkan otherwise), and offers updates when a newer llama.cpp is released.
- **Protected model server**: running models now require an API key, so other websites open in your browser can't use them. Find it in ⚙ Settings.
- **Disk-space check** before every download.
- **Bigger catalog**: models from Unsloth, bartowski, LM Studio Community and ggml-org (about 290), with a source picker when several publishers offer the same model.
- **Add any model** by pasting a HuggingFace link.
- **Gated models** (e.g. Llama) work after adding a HuggingFace token in Settings.
- **Real quality data**: live LMArena ratings for models on the public leaderboard; other models get an estimate clearly marked with ~.
