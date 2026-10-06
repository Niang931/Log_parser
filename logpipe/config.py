"""Runtime settings read from the environment (see ``.env.example``)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    logs_path: Path
    registry_path: Path
    llm_backend: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            logs_path=Path(os.environ.get("LOGS_PATH", "logs")),
            registry_path=Path(os.environ.get("REGISTRY_PATH", "artifacts/registry.json")),
            llm_backend=os.environ.get("LLM_BACKEND", "offline"),
        )
