"""Exercise the actual dashboard, manual controls, training, stop, and mobile layout.

Start the server first: uv run jev-arcade serve
Run: uv run python scripts/browser_smoke.py [--browser /usr/bin/google-chrome]
"""

import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--browser", default=None)
parser.add_argument("--screenshots", type=Path, default=Path("runs/screenshots"))
args = parser.parse_args()
args.screenshots.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=args.browser, headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(args.url.rstrip("/") + "/key-quest", wait_until="networkidle")
    expect(page.locator("#train")).to_be_enabled()
    expect(page.locator("#after-rate")).to_contain_text("%")
    page.locator("#pause").click()
    assert page.locator("#pause").get_attribute("aria-label") == "Resume playback"
    page.locator('[data-view="before"]').click()
    expect(page.locator("#view-label")).to_have_text("UNTRAINED POLICY")
    page.locator('[data-view="play"]').click()
    page.keyboard.press("ArrowUp")
    expect(page.locator("#step-label")).to_have_text("STEP 01")
    page.keyboard.press("r")
    expect(page.locator("#step-label")).to_have_text("STEP 00")

    page.locator("#episodes").fill("350")
    page.locator("#train").click()
    expect(page.locator("#stop")).to_be_visible()
    expect(page.locator("#stop")).to_be_hidden(timeout=25000)
    expect(page.locator("#after-rate")).to_have_text("100%")
    expect(page.locator("#checkpoint-slider")).to_have_attribute("max", "14")
    page.locator("#checkpoint-slider").evaluate(
        "e => { e.value = '1'; e.dispatchEvent(new Event('input')); }"
    )
    expect(page.locator("#checkpoint-title")).to_have_text("EPISODE 025")
    expect(page.locator("#view-label")).to_have_text("SAVED POLICY · EP 25")
    page.locator("#checkpoint-next").click()
    expect(page.locator("#checkpoint-title")).to_have_text("EPISODE 050")
    page.locator('[data-view="trained"]').click()
    with page.expect_download() as download:
        page.locator("#export").click()
    assert download.value.suggested_filename == "jev-demo-seed7.json"
    page.screenshot(path=str(args.screenshots / "desktop.png"), full_page=True)

    page.locator("#episodes").fill("5000")
    page.locator("#train").click()
    expect(page.locator("#stop")).to_be_visible()
    page.locator("#stop").click()
    expect(page.locator("#stop")).to_be_hidden(timeout=10000)
    expect(page.locator("#progress-caption")).to_have_text("Stopped · progress saved")

    # Never spend credits from a developer's configured live key in a browser test.
    config = page.request.get(args.url.rstrip("/") + "/api/config").json()
    if not config["jev_available"]:
        page.locator("#provider").select_option("jev")
        page.locator("#train").click()
        expect(page.locator("#error")).to_contain_text("API_KEY")
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.locator('[data-view="play"]').click()
    page.locator('[data-action="0"]').click()
    expect(page.locator("#step-label")).to_have_text("STEP 01")
    page.screenshot(path=str(args.screenshots / "mobile.png"), full_page=True)
    assert not errors, errors
    print(
        "Browser smoke passed: training, checkpoint timeline, replay, manual play, export, stop, mobile."
    )
    browser.close()
