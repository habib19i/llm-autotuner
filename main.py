import argparse
import socket
import sys
import threading
import time
import webbrowser

import httpx
import uvicorn

from backend.utils import APP_HOST, APP_NAME, APP_PORT, APP_VERSION


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((APP_HOST, port))
            return True
        except OSError:
            return False


def _is_our_app(port: int) -> bool:
    try:
        r = httpx.get(f"http://{APP_HOST}:{port}/api/health", timeout=1.5)
        return r.status_code == 200 and r.json().get("app") == APP_NAME
    except Exception:
        return False


def _open_browser_when_ready(url: str):
    for _ in range(60):
        try:
            if httpx.get(url + "api/health", timeout=0.5).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.25)
    webbrowser.open(url)


def main():
    frozen = getattr(sys, "frozen", False)
    parser = argparse.ArgumentParser(description=f"LLM Autotuner {APP_VERSION}")
    parser.add_argument("--port", type=int, default=APP_PORT, help="port for the web UI (default %(default)s)")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    parser.add_argument("--browser", action="store_true", help="open a browser window (default for the .exe)")
    args = parser.parse_args()
    open_browser = (frozen or args.browser) and not args.no_browser

    port = args.port
    if not _port_free(port):
        if _is_our_app(port):
            print(f"LLM Autotuner is already running at http://{APP_HOST}:{port}/")
            if open_browser:
                webbrowser.open(f"http://{APP_HOST}:{port}/")
            return
        for candidate in range(port + 1, port + 50):
            if _port_free(candidate):
                print(f"Port {port} is busy; using {candidate} instead.")
                port = candidate
                break
        else:
            sys.exit(f"No free port found near {port}. Use --port to choose one.")

    url = f"http://{APP_HOST}:{port}/"
    print(f"LLM Autotuner {APP_VERSION} — open {url} in your browser. Press Ctrl+C to quit.")
    if open_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    from backend.app import app
    # Reload and workers must stay off: they break PyInstaller multiprocessing
    uvicorn.run(app, host=APP_HOST, port=port, log_level="info")


if __name__ == "__main__":
    main()
