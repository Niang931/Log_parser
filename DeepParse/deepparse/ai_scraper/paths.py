"""Where the browser writes. One root, so nothing lands in whatever directory it was run from.

    <root>/                     DEEPPARSE_WEBCHAT_DIR, else DeepParse/.webchat
        profiles/<browser>/     logins; Firefox (camoufox) and Chromium profiles are not
                                interchangeable, so each browser has its own
        conversations/<site>.jsonl
        errors/<time>_<site>_<kind>/   screenshot.png, page.html, error.json

The profile always lives under the root. Conversations and errors follow ``output_dir``,
which a caller such as the eval can point into its own run directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def webchat_root() -> Path:
    if configured := os.environ.get("DEEPPARSE_WEBCHAT_DIR"):
        return Path(configured).expanduser().resolve()

    # deepparse/ai_scraper/paths.py -> the DeepParse project directory, when running from a
    # checkout (an editable install included).
    project = Path(__file__).resolve().parents[2]

    if (project / "pyproject.toml").exists():
        return project / ".webchat"

    return Path.cwd() / ".webchat"


def profile_dir(browser: str) -> Path:
    return webchat_root() / "profiles" / browser
