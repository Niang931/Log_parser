"""What can go wrong driving a chat site, sorted by what a caller should do about it.

Two flags carry the decision so callers never match on message text:

* ``retryable`` — worth another attempt in a fresh chat (a hiccup, a slow reply, a
  selector that missed because the page was mid-render).
* ``fatal`` — nothing later in this session will succeed either (signed out, bot wall,
  usage cap, browser gone). A batch run should stop rather than fail every remaining item
  the same way, slowly.

Every error raised by a session has had the page captured first — screenshot, HTML and
console — and ``artifacts`` points at that directory.
"""

from __future__ import annotations

from pathlib import Path


class WebChatError(RuntimeError):
    retryable = False
    fatal = False

    def __init__(self, message: str, *, site: str | None = None, artifacts: Path | None = None):
        super().__init__(message)
        self.message = message
        self.site = site
        self.artifacts = artifacts

    def __str__(self) -> str:
        where = f" (page captured in {self.artifacts})" if self.artifacts else ""
        return f"{self.message}{where}"


class BrowserLaunchError(WebChatError):
    """The browser did not start or could not be attached to."""

    fatal = True


class BrowserClosedError(WebChatError):
    """The browser or tab went away mid-session."""

    fatal = True


class LoginRequiredError(WebChatError):
    """The site wants a sign-in, and nobody can give it one (headless, or timed out)."""

    fatal = True


class ChallengeError(WebChatError):
    """A bot check that did not clear by itself."""

    fatal = True


class RateLimitedError(WebChatError):
    """The account hit the site's usage cap. Waiting minutes or hours is the only fix."""

    fatal = True


class ComposerNotFoundError(WebChatError):
    """The page loaded but the prompt box never appeared — a render glitch or rotted selector."""

    retryable = True


class SiteError(WebChatError):
    """The site rendered an error in place of a reply."""

    retryable = True


class ResponseTimeoutError(WebChatError):
    """The reply did not start, or did not finish, in time."""

    retryable = True
