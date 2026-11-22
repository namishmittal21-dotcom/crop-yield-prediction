"""
Start the Crop Yield Prediction website on this computer and open it in the browser.

Used by run_app.bat / run_app.sh. It runs the same Streamlit app as the online version
(streamlit_app.py) and opens http://127.0.0.1:<port> as soon as the app is ready.
"""
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))


def free_port(preferred=8501):
    """Use 8501 if it is free, otherwise any free port (e.g. if the app is already open elsewhere)."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    return preferred


def main():
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    cmd = [sys.executable, "-m", "streamlit", "run", os.path.join(HERE, "streamlit_app.py"),
           "--server.headless", "true", "--server.address", "127.0.0.1", "--server.port", str(port), "--browser.gatherUsageStats", "false"]
    app = subprocess.Popen(cmd, cwd=HERE)

    for _ in range(180):                              # wait up to 3 minutes for the server to start
        if app.poll() is not None:
            sys.exit("The app stopped unexpectedly. See the messages above.")
        try:
            urllib.request.urlopen(f"{url}/_stcore/health", timeout=1)
            break
        except OSError:
            time.sleep(1)

    print(f"\n  The app is open in your browser at {url}")
    print("  (The first visit trains the model, which takes about a minute.)")
    print("  Keep this window open while you use the app. Close it to stop the app.\n")
    webbrowser.open(url)
    try:
        app.wait()
    except KeyboardInterrupt:
        app.terminate()


if __name__ == "__main__":
    main()
