"""Verify the reward diagram, real animated gallery and accessible playback controls."""

import argparse
import io
from pathlib import Path

from PIL import Image
from playwright.sync_api import expect, sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:8000/")
parser.add_argument("--browser", default="/usr/bin/google-chrome")
args = parser.parse_args()

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=args.browser, headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1050})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(args.url, wait_until="networkidle")
    images = page.locator("#checkpoint-gallery img")
    expect(images).to_have_count(16)
    expect(page.locator(".reward-story img")).to_have_count(1)
    assert "Matched rules" not in page.locator("body").inner_text()
    assert "native return" not in page.locator("body").inner_text().lower()
    for selector in ("#reward-flow",):
        assert page.locator(selector).evaluate("e => e.complete && e.naturalWidth > 0")
    tops = [
        page.locator(s).bounding_box()["y"]
        for s in ("#results-overview", "#training-gallery", "#games")
    ]
    assert tops == sorted(tops), "Homepage sections must follow the requested order"
    expect(page.locator("#checkpoint-gallery tbody tr")).to_have_count(4)
    for src in images.evaluate_all("items => items.map(e => e.src)"):
        response = page.request.get(src)
        assert response.ok, src
        with Image.open(io.BytesIO(response.body())) as animation:
            assert animation.size == (320, 252)
            assert animation.is_animated and animation.n_frames > 1
            assert animation.info["loop"] == 0
    page.locator("#gallery-toggle").click()
    expect(page.locator("#gallery-toggle")).to_have_text("Play animations")
    expect(page.locator("#gallery-restart")).to_be_disabled()
    assert images.evaluate_all("items => items.every(e => e.src.endsWith('.png'))")
    page.locator("#gallery-toggle").click()
    expect(page.locator("#gallery-restart")).to_be_enabled()
    animated = page.locator('[data-gif="acrobot-30k.gif"]')
    animated.scroll_into_view_if_needed()
    before = animated.screenshot()
    page.wait_for_timeout(420)
    assert animated.screenshot() != before, "Gameplay must actually animate in the browser"
    page.locator("#gallery-restart").click()
    assert images.evaluate_all(
        "items => items.every(e => new URL(e.src).searchParams.get('play') === '1')"
    )
    Path("runs").mkdir(exist_ok=True)
    page.locator("#training-gallery").screenshot(path="runs/homepage-gallery-verified.png")
    page.locator("#reward-flow").screenshot(path="runs/homepage-flow-verified.png")
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    scroll = page.locator(".gallery-scroll")
    assert scroll.evaluate("e => e.scrollWidth > e.clientWidth")
    scroll.evaluate("e => e.scrollLeft = e.scrollWidth")
    assert scroll.evaluate("e => e.scrollLeft > 0")
    page.screenshot(path="runs/homepage-mobile-verified.png", full_page=True)
    assert not errors, errors
    reduced = browser.new_page(reduced_motion="reduce")
    requested_gifs = []
    reduced.on("request", lambda r: requested_gifs.append(r.url) if ".gif" in r.url else None)
    reduced.goto(args.url, wait_until="networkidle")
    expect(reduced.locator("#checkpoint-gallery img")).to_have_count(16)
    expect(reduced.locator("#gallery-toggle")).to_have_text("Play animations")
    reduced.locator("#training-gallery").scroll_into_view_if_needed()
    assert reduced.locator("#checkpoint-gallery img").evaluate_all(
        "items => items.every(e => e.src.endsWith('.png'))"
    )
    assert not requested_gifs, "Reduced motion should load only still posters"
    browser.close()
print("Homepage verified: section order, 16 real GIFs, playback, reduced motion, mobile scrolling.")
