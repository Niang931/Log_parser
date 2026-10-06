"""One chat site in one browser, driven through Playwright.

Two browsers:

``camoufox``  A patched Firefox that hides the automation fingerprints bot walls look for.
              The one to use headless.
``chrome``    The system Chromium, started as an ordinary process and attached to over the
              DevTools protocol, so the browser carries no automation flags while you sign in.

Knowing when to prompt is decided from the page, never by asking a human to press Enter.
The page is sorted into ``loading``, ``login``, ``challenge`` or ``ready``, where ready means
the composer is visible and editable and no logged-out marker is showing, twice in a row
(these apps flash a composer before redirecting to a login). In a visible window a login
or challenge is left for the human and the wait carries on by itself once it clears.
Headless, nobody can clear it, so the session gives it a few seconds (bot checks often pass
on their own) and then raises.

A reply counts as finished when a new reply exists, nothing is marked as generating, and
its text has held still for a few polls. An error banner stops the wait early.

**Tabs.** ``ask`` sends a prompt and waits for its reply, in the first tab. For several
conversations at once there is ``send`` and ``check``: ``send`` puts a prompt into a tab of
its own and returns as soon as it is submitted; ``check`` looks at that tab once and returns
the reply when it has finished, or None. Playwright's objects all belong to the thread that
opened the browser, so it is one thread taking turns between tabs — while the site writes
one reply, the others are being written too. ``deepparse.providers.web`` runs that loop.

Every failure captures the page before raising; see ``errors``.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import socket
import subprocess
import time
import traceback
import urllib.request
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeVar

from playwright.sync_api import BrowserContext, Locator, Page, Playwright, sync_playwright
from playwright.sync_api import Error as PlaywrightError

from deepparse.ai_scraper.errors import (
    BrowserClosedError,
    BrowserLaunchError,
    ChallengeError,
    ComposerNotFoundError,
    LoginRequiredError,
    RateLimitedError,
    ResponseTimeoutError,
    SiteError,
    WebChatError,
)
from deepparse.ai_scraper.paths import profile_dir, webchat_root
from deepparse.ai_scraper.sites import CHALLENGE_TITLES, RATE_LIMIT_TEXT, ChatSite

Browser = Literal["camoufox", "chrome"]
PageState = Literal["loading", "login", "challenge", "ready"]

T = TypeVar("T")

BROWSERS: tuple[Browser, ...] = ("camoufox", "chrome")


@dataclass(slots=True)
class Reply:
    text: str
    code_blocks: list[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass(slots=True)
class Pending:
    """A prompt sent into a tab, whose reply is still being written. See ``check``."""

    page: Page
    prompt: str
    new_chat: bool
    sent_at: float
    before: int
    start_by: float
    deadline: float
    timeout: float
    replying: bool = False
    last_text: str | None = None
    stable: int = 0


# Polls a reply's text must hold still for before it counts as finished.
_SETTLE_POLLS = 3


class _Timeout(Exception):
    """Internal: a poll ran out. Always converted into a WebChatError before it escapes."""


class WebChatSession:
    def __init__(
        self,
        site: ChatSite,
        *,
        browser: Browser = "camoufox",
        headless: bool = True,
        output_dir: Path | None = None,
        chrome_path: str = "chromium",
        login_timeout: float = 300,
        challenge_grace: float = 20,
        log: Callable[[str], None] = print,
    ):
        if browser not in BROWSERS:
            raise ValueError(f"Unknown browser {browser!r}; expected one of {BROWSERS}.")

        self.site = site
        self.browser = browser
        self.headless = headless
        self.output_dir = output_dir or webchat_root()
        self.chrome_path = chrome_path
        self.login_timeout = login_timeout
        self.challenge_grace = challenge_grace
        self.log = log

        self._playwright: Playwright | None = None
        self._camoufox = None
        self._chrome: subprocess.Popen | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        # Every tab, the first one included. ``_page`` is whichever one is being worked on.
        self._tabs: list[Page] = []
        self._console: deque[str] = deque(maxlen=100)

    def open(self) -> WebChatSession:
        profile = profile_dir(self.browser)
        profile.mkdir(parents=True, exist_ok=True)

        try:
            if self.browser == "camoufox":
                context = self._launch_camoufox(profile)
            else:
                context = self._launch_chrome(profile)
        except WebChatError:
            self.close()
            raise
        except Exception as error:
            self.close()
            text = str(error).strip().rstrip(".")
            if "not installed" in text or "camoufox fetch" in text:
                hint = "Download the browser with: uv run camoufox fetch"
            else:
                hint = f"If another window is using the profile at {profile}, close it first"
            raise BrowserLaunchError(
                f"Could not start {self.browser}: {text}. {hint}.", site=self.site.name
            ) from error

        self._context = context
        page = context.pages[0] if context.pages else context.new_page()
        self._watch(page)
        self._page = page
        self._tabs = [page]

        try:
            self._goto_new_chat()
            self.wait_until_ready(timeout=self.login_timeout)
        except BaseException:
            self.close()
            raise

        return self

    def close(self) -> None:
        # Each step on its own: a crashed browser must not stop the process from being reaped.
        steps: list[Callable[[], object]] = []

        if (camoufox := self._camoufox) is not None:
            steps.append(lambda: camoufox.__exit__(None, None, None))
        elif self._context is not None:
            steps.append(self._context.close)

        if self._playwright is not None:
            steps.append(self._playwright.stop)

        if self._chrome is not None:
            steps.append(self._stop_chrome)

        for step in steps:
            with contextlib.suppress(Exception):
                step()

        self._camoufox = self._playwright = self._chrome = None
        self._context = self._page = None
        self._tabs = []

    @property
    def is_open(self) -> bool:
        return bool(self._tabs) and not self._tabs[0].is_closed()

    def tab(self, index: int) -> Page:
        """The ``index``-th tab, opened (or reopened, if someone closed it) when needed."""
        if self._context is None or not self._tabs:
            raise BrowserClosedError("The browser is not open.", site=self.site.name)
        while len(self._tabs) <= index:
            self._tabs.append(self._new_tab())
        if self._tabs[index].is_closed():
            self._tabs[index] = self._new_tab()
        return self._tabs[index]

    def _new_tab(self) -> Page:
        assert self._context is not None
        try:
            page = self._context.new_page()
        except PlaywrightError as error:
            raise BrowserClosedError(
                f"Could not open another tab: {error}", site=self.site.name
            ) from error
        self._watch(page)
        return page

    def _watch(self, page: Page) -> None:
        page.on("console", lambda m: self._console.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: self._console.append(f"pageerror: {e}"))

    @contextlib.contextmanager
    def _on(self, page: Page | None) -> Iterator[None]:
        """Points every helper here at ``page`` for the duration. One thread, so it is safe."""
        if page is None:
            yield
            return
        previous, self._page = self._page, page
        try:
            yield
        finally:
            self._page = previous

    def __enter__(self) -> WebChatSession:
        return self.open()

    def __exit__(self, *_) -> None:
        self.close()

    def _launch_camoufox(self, profile: Path) -> BrowserContext:
        from camoufox.sync_api import Camoufox

        manager = Camoufox(
            persistent_context=True,
            user_data_dir=str(profile),
            headless=self.headless,
            i_know_what_im_doing=True,
        )
        # Camoufox tears itself down when __enter__ fails; only a started one is ours to close.
        context = manager.__enter__()
        self._camoufox = manager
        return context

    def _launch_chrome(self, profile: Path) -> BrowserContext:
        executable = shutil.which(self.chrome_path) or self.chrome_path
        port = _free_port()

        self._chrome = subprocess.Popen(
            [
                executable,
                f"--remote-debugging-port={port}",
                f"--user-data-dir={profile}",
                "--no-first-run",
                "--no-default-browser-check",
                *(["--headless=new"] if self.headless else []),
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        def answering() -> bool:
            if self._chrome is None or self._chrome.poll() is not None:
                raise BrowserLaunchError(
                    f"Chromium exited during startup. Is another Chromium using {profile}?",
                    site=self.site.name,
                )
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/json/version", timeout=1
                ) as response:
                    return bool(json.load(response).get("webSocketDebuggerUrl"))
            except OSError:
                return False

        try:
            self._poll(answering, timeout=30)
        except _Timeout:
            raise BrowserLaunchError(
                "Chromium's DevTools endpoint never answered.", site=self.site.name
            ) from None

        playwright = self._playwright = sync_playwright().start()
        browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
        return browser.contexts[0] if browser.contexts else browser.new_context()

    def _stop_chrome(self) -> None:
        assert self._chrome is not None
        self._chrome.terminate()
        try:
            self._chrome.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._chrome.kill()

    def page_state(self) -> PageState:
        page = self.page
        title = page.title().lower()

        if any(marker in title for marker in CHALLENGE_TITLES):
            return "challenge"
        if any(marker in page.url for marker in self.site.login_urls):
            return "login"
        if self._first_visible(self.site.logged_out):
            return "login"
        if self._composer():
            return "ready"
        return "loading"

    def wait_until_ready(self, timeout: float | None = None, quiet: bool = False) -> None:
        timeout = self.login_timeout if timeout is None else timeout
        last: PageState | None = None
        blocked_since: float | None = None
        messages = {
            "loading": f"Waiting for {self.site.name} to load...",
            "login": f"{self.site.name} wants a sign-in — complete it in the browser window.",
            "challenge": f"{self.site.name} is showing a bot check — solve it in the window.",
            "ready": f"{self.site.name} is ready.",
        }

        def ready() -> bool:
            nonlocal last, blocked_since
            state = self.page_state()

            if state != last:
                if not quiet and not (self.headless and state in ("login", "challenge")):
                    self.log(messages[state])
                last = state
                blocked_since = time.monotonic() if state in ("login", "challenge") else None

            # Headless, a sign-in can never be completed; a bot check gets a grace period
            # because they often clear on their own, and a login page a short one to settle.
            if self.headless and blocked_since is not None:
                grace = self.challenge_grace if state == "challenge" else 5
                if time.monotonic() - blocked_since > grace:
                    raise self._blocked(state)

            return state == "ready"

        try:
            self._poll(lambda: ready() and (time.sleep(1) or ready()), timeout, interval=1)
        except _Timeout:
            raise self._fail(
                ComposerNotFoundError,
                f"{self.site.name} did not show a usable prompt box within {timeout:.0f}s "
                f"(last state: {last}). The page may have changed its layout.",
            ) from None

    def _blocked(self, state: str) -> WebChatError:
        hint = (
            f"Sign in once with a visible window: python -m deepparse.ai_scraper login "
            f"{_site_key(self.site)} --browser {self.browser}"
        )
        if state == "challenge":
            return self._fail(ChallengeError, f"{self.site.name}'s bot check did not clear. {hint}")
        return self._fail(LoginRequiredError, f"{self.site.name} is not signed in. {hint}")

    def ask(self, prompt: str, *, new_chat: bool = True, timeout: float = 300) -> Reply:
        """Send ``prompt`` and return the finished reply.

        ``new_chat`` starts a fresh conversation first, so nothing from an earlier prompt
        leaks into this one. Pass False for a follow-up in the same conversation.
        """
        pending = self.send(prompt, new_chat=new_chat, timeout=timeout)
        while (reply := self.check(pending)) is None:
            time.sleep(1)
        return reply

    def send(
        self,
        prompt: str,
        *,
        new_chat: bool = True,
        timeout: float = 300,
        page: Page | None = None,
    ) -> Pending:
        """Types ``prompt`` into a tab (the first, unless ``page`` says) and submits it.

        Returns once it is sent, without waiting for the reply: ``check`` does that.
        """
        sent_at = time.monotonic()

        with self._on(page), self._converted():
            if new_chat:
                self._goto_new_chat()
            self.wait_until_ready(timeout=60, quiet=True)

            before = len(self._responses())
            composer = self._composer()
            if composer is None:
                raise self._fail(ComposerNotFoundError, "The prompt box disappeared.")

            self._type(composer, prompt)
            self._submit(composer)
            now = time.monotonic()
            return Pending(
                page=self.page,
                prompt=prompt,
                new_chat=new_chat,
                sent_at=sent_at,
                before=before,
                start_by=now + min(60, timeout),
                deadline=now + timeout,
                timeout=timeout,
            )

    def check(self, pending: Pending) -> Reply | None:
        """Looks at a sent prompt's tab once: its finished reply, or None while it is being
        written. Raises when the site shows an error or the reply runs out of time."""
        with self._on(pending.page):
            try:
                return self._check(pending)
            except WebChatError:
                raise
            except PlaywrightError as error:
                # Mid-redirect errors pass; a closed tab does not.
                if pending.page.is_closed() or "has been closed" in str(error):
                    raise BrowserClosedError(
                        f"The browser closed mid-prompt: {error}", site=self.site.name
                    ) from error
                return None

    def _check(self, pending: Pending) -> Reply | None:
        self._raise_on_error_banner()
        now = time.monotonic()

        if not pending.replying:
            if len(self._responses()) > pending.before or self._any_visible(self.site.generating):
                pending.replying = True
            elif now > pending.start_by:
                raise self._fail(ResponseTimeoutError, f"{self.site.name} never started replying.")
            else:
                return None

        responses = self._responses()
        if len(responses) > pending.before:
            last = responses[-1]
            text = last.inner_text(timeout=5_000).strip()

            if self._any_visible(self.site.generating) or not text or text != pending.last_text:
                pending.last_text, pending.stable = text, 0
            else:
                pending.stable += 1
                if pending.stable >= _SETTLE_POLLS:
                    reply = Reply(text=text, code_blocks=last.locator("pre").all_inner_texts())
                    reply.duration_ms = int((now - pending.sent_at) * 1000)
                    self._record(pending.prompt, reply, pending.new_chat)
                    return reply

        if now > pending.deadline:
            raise self._fail(
                ResponseTimeoutError,
                f"{self.site.name}'s reply did not finish within {pending.timeout:.0f}s.",
            )
        return None

    @contextlib.contextmanager
    def _converted(self) -> Iterator[None]:
        """Playwright's own errors, as the WebChatError they amount to."""
        try:
            yield
        except WebChatError:
            raise
        except PlaywrightError as error:
            if (
                not self.is_open
                or "has been closed" in str(error)
                or (self._page is not None and self._page.is_closed())
            ):
                raise BrowserClosedError(
                    f"The browser closed mid-prompt: {error}", site=self.site.name
                ) from error
            raise self._fail(SiteError, f"Browser automation failed: {error}", error) from error

    def _goto_new_chat(self) -> None:
        try:
            self.page.goto(self.site.url, wait_until="domcontentloaded", timeout=60_000)
        except PlaywrightError as error:
            raise self._fail(
                SiteError, f"Could not open {self.site.url}: {error}", error
            ) from error

    def _type(self, composer: Locator, prompt: str) -> None:
        composer.click()
        composer.evaluate(
            "(el, text) => { el.focus(); document.execCommand('insertText', false, text); }",
            prompt,
        )

        if not composer.inner_text().strip():
            composer.fill(prompt)

        if not composer.inner_text().strip():
            raise self._fail(ComposerNotFoundError, "The prompt could not be typed into the box.")

    def _submit(self, composer: Locator) -> None:
        # The send button enables a beat after the editor registers the input.
        try:
            button = self._poll(
                lambda: self._first_visible(self.site.send_button, enabled=True),
                timeout=5,
                interval=0.2,
            )
            assert button is not None
            button.click()
        except _Timeout:
            composer.press("Enter")

    def _raise_on_error_banner(self) -> None:
        banner = self._first_visible(self.site.error_banners)
        if banner is None:
            return

        text = banner.inner_text(timeout=2_000).strip()
        if not text:
            return

        if RATE_LIMIT_TEXT.search(text):
            raise self._fail(RateLimitedError, f"{self.site.name} usage limit: {text[:300]}")
        raise self._fail(SiteError, f"{self.site.name} showed an error: {text[:300]}")

    def _fail(
        self,
        kind: type[WebChatError],
        message: str,
        cause: BaseException | None = None,
    ) -> WebChatError:
        """Captures the page, then builds the error to raise. Never raises itself."""
        error = kind(message, site=self.site.name)
        error.artifacts = self.capture(kind.__name__, message, cause)
        return error

    def capture(self, label: str, message: str = "", cause: BaseException | None = None) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = self.output_dir / "errors" / f"{stamp}_{_site_key(self.site)}_{label}"
        directory.mkdir(parents=True, exist_ok=True)

        details: dict[str, object] = {
            "site": self.site.name,
            "browser": self.browser,
            "headless": self.headless,
            "error": label,
            "message": message,
            "captured_at": stamp,
            "cause": "".join(traceback.format_exception(cause)) if cause else None,
        }

        page = self._page
        if page is not None and not page.is_closed():
            for name, grab in (
                ("url", lambda: page.url),
                ("title", lambda: page.title()),
                ("page.html", lambda: (directory / "page.html").write_text(page.content())),
                ("screenshot.png", lambda: page.screenshot(path=directory / "screenshot.png")),
            ):
                try:
                    value = grab()
                    if name in ("url", "title"):
                        details[name] = value
                except Exception as error:
                    details.setdefault("capture_failures", {})[name] = str(error)  # type: ignore[index]

        details["console"] = list(self._console)
        (directory / "error.json").write_text(json.dumps(details, indent=2, default=str))
        return directory

    def _record(self, prompt: str, reply: Reply, new_chat: bool) -> None:
        path = self.output_dir / "conversations" / f"{_site_key(self.site)}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "at": datetime.now(timezone.utc).isoformat(),
            "site": self.site.name,
            "browser": self.browser,
            "new_chat": new_chat,
            "url": self.page.url,
            "duration_ms": reply.duration_ms,
            "prompt": prompt,
            "reply": reply.text,
        }

        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @property
    def page(self) -> Page:
        if self._page is None or self._page.is_closed():
            raise BrowserClosedError("The browser is not open.", site=self.site.name)
        return self._page

    def _composer(self) -> Locator | None:
        composer = self._first_visible(self.site.composer)
        return composer if composer is not None and composer.is_editable(timeout=1_000) else None

    def _responses(self) -> list[Locator]:
        for selector in self.site.responses:
            if found := self.page.locator(selector).all():
                return found
        return []

    def _first_visible(self, selectors: tuple[str, ...], enabled: bool = False) -> Locator | None:
        for selector in selectors:
            candidate = self.page.locator(selector).filter(visible=True).first
            if candidate.count() and (not enabled or candidate.is_enabled(timeout=1_000)):
                return candidate
        return None

    def _any_visible(self, selectors: tuple[str, ...]) -> bool:
        return self._first_visible(selectors) is not None

    def _poll(self, check: Callable[[], T], timeout: float, interval: float = 0.5) -> T:
        """Call ``check`` until it returns something truthy.

        Playwright errors from a page mid-redirect are retried on the next poll; a closed
        page is not, and neither is a WebChatError, which ``check`` raises on purpose.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                if result := check():
                    return result
            except PlaywrightError:
                if self._page is not None and self._page.is_closed():
                    raise

            time.sleep(interval)

        raise _Timeout


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _site_key(site: ChatSite) -> str:
    return site.name.lower()
