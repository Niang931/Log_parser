"""The scraper answers a cookie banner with "reject all", on a page shaped like Gemini's."""

from __future__ import annotations

import shutil

import pytest

from deepparse.ai_scraper.session import reject_cookies
from deepparse.ai_scraper.sites import get_site

sync_api = pytest.importorskip("playwright.sync_api")
CHROMIUM = shutil.which("chromium") or shutil.which("chromium-browser")

# Google's consent dialog: "Reject all" shows only once "Read more" expands it
PAGE = """
<div class="cdk-overlay-container"><div id="consent">
  <button onclick="document.getElementById('more').hidden = false">Read more</button>
  <div id="more" hidden>
    <button onclick="document.getElementById('consent').remove(); window.choice = 'reject'">
      <span>Reject all</span></button>
    <button onclick="window.choice = 'accept'">Accept all</button>
  </div>
</div></div>
<rich-textarea><div class="ql-editor" contenteditable="true"></div></rich-textarea>
"""


@pytest.mark.skipif(CHROMIUM is None, reason="needs chromium on PATH")
def test_cookies_are_rejected_never_accepted():
    site = get_site("gemini")
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROMIUM, headless=True)
        try:
            page = browser.new_page()
            page.set_content(PAGE)
            # first call opens "read more", the second rejects
            assert reject_cookies(page, site) is False
            assert reject_cookies(page, site) is True
            assert page.evaluate("window.choice") == "reject"
            assert page.locator("#consent").count() == 0
            # nothing left to answer
            assert reject_cookies(page, site) is False
        finally:
            browser.close()
