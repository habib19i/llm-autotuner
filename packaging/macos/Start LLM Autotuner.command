#!/bin/bash
# Double-click to start LLM Autotuner. Your browser opens automatically.
# Close this Terminal window (or press Ctrl+C) to quit.
cd "$(dirname "$0")" || exit 1
# Files downloaded from the internet are quarantined by macOS; this folder is trusted by you.
xattr -dr com.apple.quarantine . 2>/dev/null
chmod +x ./AI_Model_Autotuner
exec ./AI_Model_Autotuner
