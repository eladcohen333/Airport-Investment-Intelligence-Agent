"""Capture real screenshots of the running Streamlit app for the submission (docs/screenshots/).

Nothing is mocked: the script types questions into the live app exactly as an analyst would and saves what the app
rendered. Requires a running app and Playwright driving the locally installed Chrome or Edge (no browser download):

  python -m streamlit run app.py
  .venv\\Scripts\\python.exe -m pip install playwright
  python scripts/capture_screenshots.py --scenario all

Scenarios
  live      sidebar "Include live context" on, "What is the current operational status at LAX?"  (needs AVIATIONSTACK_API_KEY)
  ranking   "Which airports in New England are strong candidates for terminal expansion?" with "How this was calculated" open
  followup  "Compare LAX and Santa Ana Airport, SNA, congestion levels." then "What about 2024?" (context carried over)
  capacity  "What is the capacity-pressure signal at SFO and why?" (indirect proxy with its assumptions and limitations)

Exit code 0 when every requested scenario produced a screenshot, 1 otherwise. The live scenario is only reported as a
success when the rendered answer shows a retrieved Aviationstack sample; an `unavailable` state is saved but flagged.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "docs" / "screenshots"

QUESTIONS = {
    "live": ["What is the current operational status at LAX?"],
    "ranking": ["Which airports in New England are strong candidates for terminal expansion?"],
    "followup": ["Compare LAX and Santa Ana Airport, SNA, congestion levels.", "What about 2024?"],
    "capacity": ["What is the capacity-pressure signal at SFO and why?"],
}
FILES = {"live": "01_live_status_lax.png", "ranking": "02_new_england_ranking_methodology.png", "followup": "03_followup_context_2024.png",
         "capacity": "04_sfo_capacity_pressure_proxy.png"}


def _wait_for_answer(page, previous_messages: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    # the assistant turn appears as a second new chat message; wait until the spinner is gone and the app is idle
    while time.time() < deadline:
        messages = page.locator('[data-testid="stChatMessage"]').count()
        spinner = page.locator('[data-testid="stSpinner"]').count()
        running = page.locator('[data-testid="stStatusWidget"]').count()
        if messages >= previous_messages + 2 and spinner == 0 and running == 0:
            page.wait_for_timeout(2500)  # let Plotly finish drawing
            return
        page.wait_for_timeout(500)
    raise TimeoutError("the app did not finish answering in time")


def _ask(page, question: str, timeout_s: float) -> None:
    before = page.locator('[data-testid="stChatMessage"]').count()
    box = page.locator('textarea[data-testid="stChatInputTextArea"]')
    box.click()
    box.fill(question)
    box.press("Enter")
    _wait_for_answer(page, before, timeout_s)


def _fit_viewport(page, width: int) -> None:
    """Grow the viewport to the main column's full height so one screenshot shows the whole conversation."""
    for _ in range(3):
        # Streamlit's outer containers are overflow:hidden at viewport height; the block container is the real content height
        needed = page.evaluate(
            "() => { const m = document.querySelector('[data-testid=\"stMainBlockContainer\"]');"
            " const b = document.querySelector('[data-testid=\"stBottom\"]');"
            " return Math.ceil((m ? m.scrollHeight : document.body.scrollHeight) + (b ? b.clientHeight : 0)); }"
        )
        height = max(900, min(int(needed) + 80, 9000))
        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_timeout(800)


def run_scenario(browser, base_url: str, name: str, out_dir: Path, width: int, timeout_s: float, scale: int) -> tuple[Path, str]:
    context = browser.new_context(viewport={"width": width, "height": 1000}, device_scale_factor=scale)
    page = context.new_page()
    try:
        page.goto(base_url, wait_until="networkidle")
        page.wait_for_selector('textarea[data-testid="stChatInputTextArea"]', timeout=60_000)
        page.wait_for_timeout(1500)
        if name == "live":
            page.get_by_text("Include live context", exact=True).click()
            page.wait_for_timeout(1500)
        for q in QUESTIONS[name]:
            _ask(page, q, timeout_s)
        if name in ("ranking", "capacity"):
            page.get_by_text("How this was calculated", exact=True).last.click()
            page.wait_for_timeout(1200)
        _fit_viewport(page, width)
        body = page.locator('[data-testid="stChatMessage"]').last.inner_text()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / FILES[name]
        page.screenshot(path=str(path), full_page=False)
        return path, body
    finally:
        context.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--scenario", choices=["live", "ranking", "followup", "capacity", "all"], default="all")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--scale", type=int, default=2, help="device scale factor (2 = retina-quality PNG)")
    ap.add_argument("--timeout", type=float, default=180.0, help="seconds to wait for each answer")
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    args = ap.parse_args(argv)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed: .venv\\Scripts\\python.exe -m pip install playwright")
        return 1

    names = ["live", "ranking", "followup", "capacity"] if args.scenario == "all" else [args.scenario]
    ok = True
    with sync_playwright() as p:
        browser = None
        for channel in ("chrome", "msedge"):
            try:
                browser = p.chromium.launch(channel=channel, headless=not args.headed)
                break
            except Exception as exc:  # noqa: BLE001 - try the next installed browser
                print(f"could not launch {channel}: {str(exc).splitlines()[0]}")
        if browser is None:
            print("No installed Chrome or Edge could be launched.")
            return 1
        try:
            for name in names:
                try:
                    path, body = run_scenario(browser, args.url, name, Path(args.out), args.width, args.timeout, args.scale)
                except Exception as exc:  # noqa: BLE001 - report and continue with the other scenarios
                    print(f"[{name}] FAILED: {type(exc).__name__}: {str(exc).splitlines()[0]}")
                    ok = False
                    continue
                verdict = "saved"
                if name == "live":
                    low = body.lower()
                    if "aviationstack" in low and "unavailable" not in low and "live retrieved at: n/a" not in low:
                        verdict = "saved (live sample retrieved from Aviationstack)"
                    else:
                        verdict = "saved, but the live sample was NOT retrieved (key missing, quota, or provider error) - do not use as API proof"
                        ok = False
                print(f"[{name}] {verdict}: {path}")
        finally:
            browser.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
