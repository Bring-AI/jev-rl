"""Exercise all classic replay controls and a real new training run, without paid API calls."""

import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:8002")
parser.add_argument("--browser", default="/usr/bin/google-chrome")
parser.add_argument("--static", action="store_true")
args = parser.parse_args()
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=args.browser, headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(args.url, wait_until="networkidle")
    expect(page.locator("#success")).to_have_text("100%")
    for game in ("cartpole", "mountaincar", "acrobot", "frozenlake"):
        page.locator(f'[data-game="{game}"]').click()
        expect(page.locator("#replay-label")).to_contain_text("SAVED POLICY", timeout=15000)
        for provider in ("jev", "native", "rules"):
            page.locator("#recorded-provider").select_option(provider)
            page.locator("#recorded-seed").select_option("19")
            expect(page.locator("#native-return")).not_to_have_text("—")
            page.locator("#before").click()
            expect(page.locator("#replay-label")).to_have_text("UNTRAINED POLICY")
            page.locator("#after").click()
            expect(page.locator("#replay-label")).to_contain_text("SAVED POLICY")
        page.locator("#recorded-provider").select_option("jev")
        page.locator("#recorded-seed").select_option("7")
    page.locator("#play-pause").click()
    expect(page.locator("#play-pause")).to_have_attribute("aria-label", "Resume playback")
    page.locator("#restart").click()
    expect(page.locator("#frame-number")).to_have_text("STEP 0")
    with page.expect_download() as info:
        page.locator("#download").click()
    assert info.value.suggested_filename == "jevrl-frozenlake-jev-seed7.json"
    if args.static:
        expect(page.locator("#train")).to_be_disabled()
    else:
        # Official saved JEV responses, no new model requests.
        page.locator("#train-provider").select_option("recorded")
        page.locator("#train-steps").fill("256")
        page.locator("#save-every").fill("128")
        page.locator("#train").click()
        expect(page.locator("#training-status")).to_contain_text("completed", timeout=30000)
        expect(page.locator("#checkpoint")).to_have_attribute("max", "2")
        result = page.request.get(args.url + "/api/classic/export").json()
        assert result["provider"] == "jev"
        assert result["reward"]["current_session"]["api_calls"] == 0
        page.locator("#train-steps").fill("1000000")
        page.locator("#train-provider").select_option("rules")
        page.locator("#train").click()
        expect(page.locator("#stop")).to_be_visible()
        page.locator("#stop").click()
        expect(page.locator("#training-status")).to_contain_text("stopped", timeout=30000)
    page.locator('[data-game="cartpole"]').click()
    expect(page.locator("#success")).to_have_text("100%")
    Path("assets").mkdir(exist_ok=True)
    page.locator(".workspace").screenshot(path="assets/classic-dashboard.png")
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.screenshot(path="runs/classic-mobile-verified.png", full_page=True)
    assert not errors, errors
    browser.close()
print(
    "Classic browser smoke passed: four games, three reward sources, seeds, checkpoints, playback, downloads, training/stop, mobile."
)
