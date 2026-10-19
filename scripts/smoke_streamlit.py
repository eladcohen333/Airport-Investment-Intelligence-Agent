"""Start the Streamlit app headless on a spare port, wait for its health endpoint, then stop it.

Usage: python scripts/smoke_streamlit.py [--port 8765] [--timeout 90]
Exit code 0 when /_stcore/health answers 200 (the app imported and started), 1 otherwise.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--timeout", type=int, default=90)
    args = ap.parse_args()
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true",
           "--server.port", str(args.port), "--browser.gatherUsageStats", "false"]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    ok = False
    deadline = time.time() + args.timeout
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://localhost:{args.port}/_stcore/health", timeout=2) as resp:
                    if resp.status == 200:
                        ok = True
                        break
            except Exception:
                time.sleep(1)
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
    tail = "\n".join((out or "").strip().splitlines()[-8:])
    print(f"STREAMLIT HEALTH: {'OK' if ok else 'FAILED'} (port {args.port})")
    if tail:
        print("--- streamlit output (tail) ---")
        print(tail)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
