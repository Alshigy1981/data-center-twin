"""
DC Twin Desktop Launcher
════════════════════════
Single entry point for the packaged desktop application.

Start-up sequence
─────────────────
1. FastAPI / uvicorn starts in a daemon thread.
2. Streamlit starts in a daemon multiprocessing.Process.
3. A PyWebView window opens immediately with a dark splash screen.
4. A background thread polls TCP ports 8000 and 8501.
5. Once both are ready the window navigates to http://127.0.0.1:8501.

PyInstaller compatibility
─────────────────────────
multiprocessing.freeze_support() is called at the very top of main() so
the spawned Streamlit child-process is correctly intercepted by PyInstaller
and calls _streamlit_worker() instead of re-running main().
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import socket
import sys
import threading
import time
from pathlib import Path

# ─────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────

if getattr(sys, "frozen", False):
    # Running inside a PyInstaller bundle
    BUNDLE_DIR: Path = Path(sys._MEIPASS)          # type: ignore[attr-defined]
else:
    BUNDLE_DIR = Path(__file__).parent

DASHBOARD_SCRIPT = str(BUNDLE_DIR / "dashboard" / "app.py")

# Persistent data directory — survives app updates
if sys.platform == "darwin":
    _APP_SUPPORT = Path.home() / "Library" / "Application Support" / "DCTwin"
elif sys.platform == "win32":
    _APP_SUPPORT = Path(os.getenv("APPDATA", str(Path.home()))) / "DCTwin"
else:
    _APP_SUPPORT = Path.home() / ".dc_twin"

_APP_SUPPORT.mkdir(parents=True, exist_ok=True)
DB_PATH = str(_APP_SUPPORT / "data_center.db")

BACKEND_PORT = 8000
DASHBOARD_PORT = 8501
STARTUP_TIMEOUT = 90  # seconds to wait for services

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("dc-twin")


# ─────────────────────────────────────────
# Subprocess workers
# ─────────────────────────────────────────

def _fastapi_worker() -> None:
    """Run uvicorn (FastAPI) in the calling thread."""
    # Ensure bundle packages are importable when frozen
    if str(BUNDLE_DIR) not in sys.path:
        sys.path.insert(0, str(BUNDLE_DIR))

    os.environ.setdefault("DB_PATH", DB_PATH)
    os.environ.setdefault("API_BASE_URL", f"http://127.0.0.1:{BACKEND_PORT}")
    os.environ.setdefault("SIMULATION_INTERVAL", "30")

    import uvicorn
    from backend.main import app  # noqa: F401 — import after path is set

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=BACKEND_PORT,
        log_level="warning",
        access_log=False,
    )


def _streamlit_worker(
    script_path: str,
    db_path: str,
    bundle_dir: str,
    backend_port: int,
    dashboard_port: int,
) -> None:
    """
    Streamlit server — runs in a separate process.
    PyInstaller intercepts the spawn and calls this function directly
    rather than re-running main(), because freeze_support() is in place.
    """
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)

    os.environ["DB_PATH"] = db_path
    os.environ["API_BASE_URL"] = f"http://127.0.0.1:{backend_port}"

    from streamlit.web import bootstrap  # type: ignore[import-untyped]

    bootstrap.run(
        script_path,
        "",           # command_line (empty)
        [],           # args
        {
            "server.port": dashboard_port,
            "server.address": "127.0.0.1",
            "server.headless": True,
            "browser.gatherUsageStats": False,
            "global.developmentMode": False,
            "logger.level": "warning",
        },
    )


# ─────────────────────────────────────────
# Port readiness check
# ─────────────────────────────────────────

def _wait_for_port(port: int, timeout: float = STARTUP_TIMEOUT) -> bool:
    """Block until TCP port accepts connections or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.4)
    return False


# ─────────────────────────────────────────
# Splash screen HTML
# ─────────────────────────────────────────

_SPLASH_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f1923;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    color: #c8d8e8;
    user-select: none;
  }
  .icon { font-size: 4rem; margin-bottom: 20px; line-height: 1; }
  h1 {
    font-size: 1.7rem;
    font-weight: 700;
    color: #e0f0ff;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
  }
  .sub {
    font-size: 0.85rem;
    color: #3a5a7a;
    margin-bottom: 40px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .spinner {
    width: 36px; height: 36px;
    border: 3px solid #1a3050;
    border-top-color: #00aaff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 18px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .status {
    font-size: 0.72rem;
    color: #2a5070;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .version {
    position: absolute;
    bottom: 18px;
    font-size: 0.65rem;
    color: #1a3050;
    letter-spacing: 0.06em;
  }
</style>
</head>
<body>
  <div class="icon">⚡</div>
  <h1>Data Center Digital Twin</h1>
  <div class="sub">Operations Center</div>
  <div class="spinner"></div>
  <div class="status" id="status">Starting services…</div>
  <div class="version">v1.0.0 · Local Edition</div>
  <script>
    const msgs = [
      'Starting FastAPI backend…',
      'Initialising database…',
      'Starting Streamlit dashboard…',
      'Loading telemetry engine…',
      'Almost ready…',
    ];
    let i = 0;
    const el = document.getElementById('status');
    setInterval(() => {
      el.textContent = msgs[Math.min(i++, msgs.length - 1)];
    }, 2800);
  </script>
</body>
</html>"""


# ─────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────

def main() -> None:
    """Start all services and open the PyWebView desktop window."""

    # MUST be the first call in main() for PyInstaller + multiprocessing
    multiprocessing.freeze_support()

    # Ensure bundle is importable (non-frozen dev mode)
    if str(BUNDLE_DIR) not in sys.path:
        sys.path.insert(0, str(BUNDLE_DIR))

    log.info("DC Twin starting up — bundle=%s  db=%s", BUNDLE_DIR, DB_PATH)

    # ── 1. FastAPI in a daemon thread ──────────────────────────────────────
    api_thread = threading.Thread(
        target=_fastapi_worker, daemon=True, name="dc-twin-fastapi"
    )
    api_thread.start()
    log.info("FastAPI thread started on port %d", BACKEND_PORT)

    # ── 2. Streamlit in a daemon process ───────────────────────────────────
    st_proc = multiprocessing.Process(
        target=_streamlit_worker,
        args=(
            DASHBOARD_SCRIPT,
            DB_PATH,
            str(BUNDLE_DIR),
            BACKEND_PORT,
            DASHBOARD_PORT,
        ),
        daemon=True,
        name="dc-twin-streamlit",
    )
    st_proc.start()
    log.info("Streamlit process started (pid=%d) on port %d", st_proc.pid, DASHBOARD_PORT)

    # ── 3. PyWebView window with splash ────────────────────────────────────
    import webview  # imported here so freeze_support() above fires first

    window = webview.create_window(
        title="DC Twin | Operations Center",
        html=_SPLASH_HTML,
        width=1440,
        height=900,
        min_size=(1024, 700),
        background_color="#0f1923",
        text_select=False,
    )

    def _load_when_ready(win: "webview.Window") -> None:
        """Background thread: wait for services, then navigate the window."""
        log.info("Waiting for FastAPI (port %d)…", BACKEND_PORT)
        if not _wait_for_port(BACKEND_PORT):
            log.error("FastAPI did not start within %ds — aborting", STARTUP_TIMEOUT)
            win.load_html("<h1 style='color:red;font-family:sans-serif;padding:40px'>"
                          "⚠ FastAPI failed to start. Please restart the app.</h1>")
            return

        log.info("FastAPI ready. Waiting for Streamlit (port %d)…", DASHBOARD_PORT)
        if not _wait_for_port(DASHBOARD_PORT):
            log.error("Streamlit did not start within %ds — aborting", STARTUP_TIMEOUT)
            win.load_html("<h1 style='color:red;font-family:sans-serif;padding:40px'>"
                          "⚠ Dashboard failed to start. Please restart the app.</h1>")
            return

        # Brief pause so Streamlit's React frontend finishes mounting
        time.sleep(2.0)
        log.info("All services ready — loading dashboard")
        win.load_url(f"http://127.0.0.1:{DASHBOARD_PORT}")

    threading.Thread(
        target=_load_when_ready, args=(window,), daemon=True, name="dc-twin-waiter"
    ).start()

    # webview.start() blocks until the window is closed (must run on main thread)
    webview.start(debug=False)

    # ── Graceful shutdown ──────────────────────────────────────────────────
    log.info("Window closed — shutting down")
    if st_proc.is_alive():
        st_proc.terminate()
        st_proc.join(timeout=3)
    log.info("DC Twin exited cleanly")


if __name__ == "__main__":
    main()
