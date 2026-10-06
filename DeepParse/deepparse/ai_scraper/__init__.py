"""Driving the Gemini, Claude and ChatGPT web apps from a signed-in browser.

For synthesising masks without an API key; see ``deepparse.providers.web`` and
``deepparse synth --mode llm --provider web``.
"""

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
from deepparse.ai_scraper.paths import webchat_root
from deepparse.ai_scraper.session import BROWSERS, Browser, Pending, Reply, WebChatSession
from deepparse.ai_scraper.sites import SITES, ChatSite, get_site

__all__ = [
    "BROWSERS",
    "SITES",
    "Browser",
    "BrowserClosedError",
    "BrowserLaunchError",
    "ChallengeError",
    "ChatSite",
    "ComposerNotFoundError",
    "LoginRequiredError",
    "Pending",
    "RateLimitedError",
    "Reply",
    "ResponseTimeoutError",
    "SiteError",
    "WebChatError",
    "WebChatSession",
    "get_site",
    "webchat_root",
]
