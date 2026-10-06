"""``python -m deepparse.ai_scraper`` — sign in once, check a session, or send a prompt.

login  <site>            visible window; waits for you to sign in, then saves the profile
check  <site>            headless; exit 0 if the saved login still works, 1 if not
ask    <site> <prompt>   prints the reply
"""

from __future__ import annotations

import argparse
import sys

from deepparse.ai_scraper.errors import WebChatError
from deepparse.ai_scraper.paths import webchat_root
from deepparse.ai_scraper.session import BROWSERS, WebChatSession
from deepparse.ai_scraper.sites import SITES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m deepparse.ai_scraper", description="Drive a signed-in chat web app."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("site", choices=sorted(SITES))
        sub.add_argument("--browser", choices=BROWSERS, default="camoufox")
        sub.add_argument("--chrome-path", default="chromium")

    login = commands.add_parser("login", help="Sign in with a visible window.")
    common(login)
    login.add_argument("--timeout", type=float, default=600)

    check = commands.add_parser("check", help="Confirm the saved login works headless.")
    common(check)

    ask = commands.add_parser("ask", help="Send one prompt and print the reply.")
    common(ask)
    ask.add_argument("prompt")
    ask.add_argument("--headed", action="store_true", help="Show the browser window.")
    ask.add_argument("--timeout", type=float, default=300)

    args = parser.parse_args(argv)
    log = lambda message: print(message, file=sys.stderr)  # noqa: E731

    session = WebChatSession(
        SITES[args.site],
        browser=args.browser,
        headless=args.command == "check" or (args.command == "ask" and not args.headed),
        chrome_path=args.chrome_path,
        login_timeout=args.timeout if args.command == "login" else 60,
        log=log,
    )

    try:
        with session:
            if args.command == "ask":
                print(session.ask(args.prompt, new_chat=False, timeout=args.timeout).text)
            elif args.command == "login":
                log(f"Signed in. Profile saved under {webchat_root() / 'profiles'}.")
    except WebChatError as error:
        log(f"{type(error).__name__}: {error}")
        return 1
    except KeyboardInterrupt:
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
