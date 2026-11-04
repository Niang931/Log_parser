"""``logpipe server``: agent intake, the operator UI's API and the learning lifecycle."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path


def run(*, host: str, port: int, store: str, queue: str, llm: str, browser: str, data_dir: Path,
        epochs: int, demo: bool = False) -> None:
    import uvicorn

    from ..enrich.llm import get_adapter
    from ..storage import open_store
    from ..unresolved import open_queue
    from .app import create_app
    from .control import ClickHouseDocs, ControlStore
    from .core import Server

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log_store = open_store(store)
    # shortcut: with the memory store the control plane is in memory too; use clickhouse to keep agents, keys and models across restarts.
    control = ControlStore(ClickHouseDocs(log_store.client) if store == "clickhouse" else None)  # type: ignore[attr-defined]
    server = Server(log_store, control, data_dir=data_dir, epochs=epochs,
                    adapter_factory=lambda: get_adapter(llm, browser=browser),  # type: ignore[arg-type]
                    queue_factory=lambda: open_queue(queue),
                    bootstrap_key=os.environ.get("LOGPIPE_BOOTSTRAP_KEY") or None)
    demo_instance = None
    if demo:
        from ..demo.server import Demo

        demo_instance = Demo(llm=llm, browser=browser, cache_dir=Path("artifacts/demo"))  # type: ignore[arg-type]
        threading.Thread(target=demo_instance.replay, daemon=True).start()
    print(f"[server] logpipe on http://{host}:{port}/ ({store} store, {queue} queue, {llm})", flush=True)
    try:
        uvicorn.run(create_app(server, demo_instance), host=host, port=port, log_level="warning")
    finally:
        server.close()
