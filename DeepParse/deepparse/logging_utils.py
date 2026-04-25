"""Logging utilities for structured artifact output."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

try:  # Rich is optional; fall back to basic StreamHandler when unavailable
    from rich.logging import RichHandler
except Exception:  # pragma: no cover - exercised when rich is absent
    class RichHandler(logging.StreamHandler):  # type: ignore[no-redef]
        """Fallback handler mimicking :class:`rich.logging.RichHandler` signature."""

        def __init__(self, *args, **kwargs) -> None:
            kwargs.pop("rich_tracebacks", None)
            kwargs.pop("markup", None)
            super().__init__(*args, **kwargs)


def configure_logging(log_dir: str, log_name: str) -> Path:
    """Setup logging handlers writing to file and stdout.

    Args:
        log_dir: Directory where logs should be stored.
        log_name: Base filename for the log file.

    Returns:
        Path to the log file.
    """

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{log_name}.log"

    handlers: list[logging.Handler] = [RichHandler(rich_tracebacks=True, markup=True)]

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info("Logging initialised. Writing to %s", os.fspath(log_path))
    return log_path


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-specific logger."""

    return logging.getLogger(name)
