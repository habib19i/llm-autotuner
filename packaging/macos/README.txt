LLM Autotuner for macOS (Apple Silicon: M1, M2, M3, M4 …)
==========================================================

Start it
--------
Double-click "Start LLM Autotuner.command". Your browser opens the app.
Keep the Terminal window open while you use it; close it to quit.

If macOS says the app "cannot be opened" or "Apple could not verify" it
---------------------------------------------------------------------
This happens because the app is not (yet) signed with a paid Apple certificate.

Option A (easiest):
  1. Open System Settings > Privacy & Security.
  2. Scroll down: next to the message about "Start LLM Autotuner.command",
     click "Open Anyway", then confirm.

Option B (Terminal, one time):
  Open the Terminal app and run (adjust the path if you moved the folder):

      xattr -dr com.apple.quarantine ~/Downloads/"LLM Autotuner"

  Then double-click "Start LLM Autotuner.command" again.

Where things are stored
-----------------------
Models, the llama.cpp engine, and settings are kept in this folder
(models/, bin/, cache/, logs/). Move the whole folder if you want them elsewhere,
for example to an external drive.

Intel Mac?
----------
This download is for Apple Silicon. On an Intel Mac, run the app from source:
see "Run from source" in the README at https://github.com/habib19i/llm-autotuner
