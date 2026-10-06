"""The chat sites, as data. Everything that differs between them lives here.

Selectors are the part that rots. Each field keeps several, most specific first, and the
first one that matches wins — so a site redesign usually costs one new selector at the front
of a tuple rather than a code change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChatSite:
    name: str
    # Opening this starts a fresh conversation.
    url: str
    # The editable prompt box.
    composer: tuple[str, ...]
    send_button: tuple[str, ...]
    # Present only while a reply is streaming.
    generating: tuple[str, ...]
    # One element per assistant reply, in document order.
    responses: tuple[str, ...]
    # URL fragments that mean "you are being asked to sign in".
    login_urls: tuple[str, ...]
    # Elements that only exist for a logged-out visitor (some sites show a composer anyway).
    logged_out: tuple[str, ...] = ()
    # Elements the site uses to show a failed reply.
    error_banners: tuple[str, ...] = ()


SITES: dict[str, ChatSite] = {
    "gemini": ChatSite(
        name="Gemini",
        url="https://gemini.google.com/app",
        composer=("rich-textarea div.ql-editor[contenteditable='true']", "div.ql-editor"),
        send_button=("button.send-button", "button[aria-label*='Send']"),
        generating=("button[aria-label*='Stop']", ".is-generating"),
        responses=("model-response message-content", "message-content", ".model-response-text"),
        login_urls=("accounts.google.com", "/ServiceLogin", "/signin"),
        logged_out=("a[href*='ServiceLogin']", "a[aria-label='Sign in']"),
        error_banners=("model-response .error-message", "snack-bar-container"),
    ),
    "claude": ChatSite(
        name="Claude",
        url="https://claude.ai/new",
        composer=(
            "div[contenteditable='true'].ProseMirror",
            "[data-testid='chat-input'] [contenteditable='true']",
        ),
        send_button=("button[aria-label='Send message']", "button[aria-label*='Send']"),
        generating=("[data-is-streaming='true']", "button[aria-label='Stop response']"),
        responses=("div.font-claude-response", "[data-is-streaming]"),
        login_urls=("claude.ai/login", "/magic-link", "accounts.google.com"),
        error_banners=("[data-testid='message-warning']", "[role='alert']"),
    ),
    "chatgpt": ChatSite(
        name="ChatGPT",
        url="https://chatgpt.com/",
        composer=("#prompt-textarea[contenteditable='true']", "#prompt-textarea"),
        send_button=("button[data-testid='send-button']", "#composer-submit-button"),
        generating=("button[data-testid='stop-button']",),
        responses=("div[data-message-author-role='assistant']",),
        login_urls=("auth.openai.com", "/auth/login", "accounts.google.com"),
        logged_out=("button[data-testid='login-button']",),
        error_banners=(".text-token-text-error", "[role='alert']"),
    ),
}

# Titles shown by interstitial bot checks (Cloudflare and friends).
CHALLENGE_TITLES = ("just a moment", "attention required", "verify you are human")

# Text in an error banner that means the account is capped, not that the reply glitched.
RATE_LIMIT_TEXT = re.compile(
    r"usage limit|reached (?:your|our|the) (?:\w+ )?limit|too many (?:requests|messages)"
    r"|out of (?:free )?messages|try again (?:later|in \d|after)",
    re.IGNORECASE,
)


def get_site(name: str) -> ChatSite:
    try:
        return SITES[name.strip().lower()]
    except KeyError:
        raise ValueError(
            f"No chat site named {name!r}. Known sites: {', '.join(sorted(SITES))}."
        ) from None
