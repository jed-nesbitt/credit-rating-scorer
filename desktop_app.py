import sys
import time
import urllib.request
import multiprocessing as mp
from pathlib import Path
import socket
import traceback

import webview

# Force PyInstaller to include your code (VERY IMPORTANT)
import main  # noqa: F401
import credit_engine  # noqa: F401
PORT = 8501
URL = "http://127.0.0.1:8501"


def resource_path(relative_path: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative_path


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(url: str, timeout_seconds: int = 60) -> bool:
    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def run_streamlit_server(port: int):
    # Runs in a child PROCESS (main thread there) -> avoids signal/thread issues
    from streamlit.web import cli as stcli

    log_dir = Path.home() / "CreditRiskDesktopLogs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "streamlit_crash.log"

    try:
        app_path = resource_path("app.py")

        sys.argv = [
    "streamlit",
    "run",
    str(app_path),
    "--server.headless=true",
    "--server.address=127.0.0.1",
    "--browser.gatherUsageStats=false",
    "--global.developmentMode=false",
        ]


        stcli.main()

    except SystemExit:
        pass
    except Exception:
        log_file.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main_entry():
    mp.freeze_support()

    port = get_free_port()
    url = f"http://127.0.0.1:{port}"

    p = mp.Process(target=run_streamlit_server, args=(port,), daemon=True)
    p.start()

    if not wait_for_server(url, timeout_seconds=60):
        # If Streamlit crashed, a log will be written here:
        log_path = Path.home() / "CreditRiskDesktopLogs" / "streamlit_crash.log"
        raise RuntimeError(
            "Streamlit server did not start. "
            f"Check log: {log_path}"
        )

    webview.create_window("Credit Risk Engine", url, width=1200, height=800)
    webview.start()

    if p.is_alive():
        p.terminate()
        p.join(3)


if __name__ == "__main__":
    main_entry()
