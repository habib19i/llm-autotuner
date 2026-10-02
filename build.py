import os
import subprocess
import sys

# IMPORTANT: run this from a CLEAN environment that has only requirements.txt +
# pyinstaller installed (e.g. `.buildvenv`, see below) — not a general-purpose
# Anaconda "base" environment. A base env with unrelated packages installed
# (e.g. both PyQt5 and PySide6 pulled in by Jupyter/Spyder) will make
# PyInstaller's dependency analysis abort with "attempt to collect multiple Qt
# bindings packages". To (re)create the clean build env from the project root:
#
#   <python> -m venv .buildvenv
#   .buildvenv\Scripts\python.exe -m pip install -r requirements-dev.txt
#   .buildvenv\Scripts\python.exe build.py            (add --skip-tests to skip pytest)
#
# If <python> is a conda/Anaconda interpreter, the venv's stdlib C-extensions
# (_ctypes, _decimal, _bz2, _lzma, pyexpat) depend on DLLs (ffi.dll,
# libexpat.dll, etc.) that live in <conda-root>\Library\bin, which isn't on
# PATH by default outside the base env. Add it before building, e.g.:
#
#   $env:PATH = "<conda-root>\Library\bin;" + $env:PATH
#
# Otherwise PyInstaller silently fails to bundle those DLLs and the resulting
# .exe crashes on startup with "DLL load failed while importing _ctypes".

ROOT = os.path.dirname(os.path.abspath(__file__))


def build():
    os.chdir(ROOT)
    if "--skip-tests" not in sys.argv:
        print("Running tests...")
        subprocess.check_call([sys.executable, "-m", "pytest", "-q"])

    print("Building executable with PyInstaller...")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "AI_Model_Autotuner",
        "--onefile",
        "--clean",
        "--noconfirm",
        "--add-data", f"frontend{os.pathsep}frontend",
        "--add-data", f"{os.path.join('backend', 'data')}{os.pathsep}{os.path.join('backend', 'data')}",
        "--collect-submodules", "backend",
        "main.py"
    ]

    # macOS: sign with a Developer ID when one is configured (see .github/workflows/release.yml)
    identity = os.environ.get("MACOS_CODESIGN_IDENTITY")
    if sys.platform == "darwin" and identity:
        cmd.extend(["--codesign-identity", identity, "--osx-entitlements-file",
                    os.path.join("packaging", "entitlements.plist")])

    # Uvicorn picks its protocol/loop implementations dynamically, so PyInstaller misses them
    cmd.extend([
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.http.h11_impl",
        "--hidden-import", "uvicorn.protocols.http.httptools_impl",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.protocols.websockets.websockets_impl",
        "--hidden-import", "uvicorn.protocols.websockets.wsproto_impl",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
        # Test-only / unused heavy modules
        "--exclude-module", "pytest",
        "--exclude-module", "tkinter",
    ])

    subprocess.check_call(cmd)
    exe = "AI_Model_Autotuner.exe" if os.name == "nt" else "AI_Model_Autotuner"
    print(f"Build complete! Check the 'dist' folder for {exe}")


if __name__ == "__main__":
    build()
