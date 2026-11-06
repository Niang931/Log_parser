"""``logpipe server``: agent intake, the operator UI's API and the learning lifecycle."""
from __future__ import annotations

import logging
import logging.handlers
import os
import threading
from pathlib import Path


class _Console(logging.Formatter):
    """One line per record; tracebacks only go to the log file."""

    def __init__(self, log_file: Path):
        super().__init__(datefmt="%H:%M:%S")
        self.log_file = log_file

    def format(self, record: logging.LogRecord) -> str:
        line = f"{self.formatTime(record, self.datefmt)} {record.levelname.lower()} {record.getMessage()}"
        if record.exc_info and record.exc_info[1] is not None:
            line += f": {record.exc_info[1]} (details in {self.log_file})"
        return line


class _Json(logging.Formatter):
    """One JSON object per line (``LOGPIPE_LOG_FORMAT=json``), for log shippers, logpipe's own
    agent included."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        out = {"ts": self.formatTime(record), "level": record.levelname.lower(), "logger": record.name,
               "message": record.getMessage(), "thread": record.threadName}
        if record.exc_info:
            out["error"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging(data_dir: Path) -> Path:
    """Everything, with tracebacks, to ``data_dir/server.log``; warnings and errors as one
    line each to the console."""
    data_dir.mkdir(parents=True, exist_ok=True)
    log_file = data_dir / "server.log"
    to_file = logging.handlers.RotatingFileHandler(log_file, maxBytes=10 << 20, backupCount=3)
    to_file.setFormatter(_Json() if os.environ.get("LOGPIPE_LOG_FORMAT") == "json"
                         else logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(_Console(log_file))
    logging.basicConfig(level=logging.INFO, handlers=[to_file, console], force=True)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # every heartbeat and long-poll
    logging.captureWarnings(True)  # library warnings (torch, sklearn) go to the file too
    return log_file


def run(*, host: str, port: int, store: str, queue: str, llm: str, browser: str, data_dir: Path,
        epochs: int, demo: bool = False) -> None:
    import uvicorn

    from ..enrich.llm import get_adapter
    from ..storage import open_store
    from ..unresolved import open_queue
    from .app import create_app
    from .control import ClickHouseDocs, ControlStore
    from .core import Server

    log_file = setup_logging(data_dir)
    log_store = open_store(store)
    # shortcut: with the memory store the control plane is in memory too; use clickhouse to keep agents, keys and models across restarts.
    control = ControlStore(ClickHouseDocs(log_store.client) if store == "clickhouse" else None)  # type: ignore[attr-defined]
    server = Server(log_store, control, data_dir=data_dir, epochs=epochs,
                    adapter_factory=lambda: get_adapter(llm, browser=browser),  # type: ignore[arg-type]
                    queue_factory=lambda: open_queue(queue),
                    bootstrap_key=os.environ.get("LOGPIPE_BOOTSTRAP_KEY") or None)
    from .auth import Auth

    generated = Auth(control).ensure(os.environ.get("LOGPIPE_ADMIN_PASSWORD") or None)
    demo_instance = None
    if demo:
        from ..demo.server import Demo

        demo_instance = Demo(llm=llm, browser=browser, cache_dir=Path("artifacts/demo"))  # type: ignore[arg-type]
        threading.Thread(target=demo_instance.replay, daemon=True).start()
    print(f"[server] logpipe on http://{host}:{port}/ ({store} store, {queue} queue, {llm}); "
          f"log: {log_file}", flush=True)
    if generated:
        print(f"[server] admin password for the UI: {generated}  (change it in the UI; set "
              "LOGPIPE_ADMIN_PASSWORD to reset it)", flush=True)
    try:
        uvicorn.run(create_app(server, demo_instance), host=host, port=port, log_config=None,
                    timeout_graceful_shutdown=2)  # SSE streams and agent long-polls never end on their own
    finally:
        server.close()
