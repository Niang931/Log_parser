"""Minimal YAML loader with optional PyYAML dependency."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # Optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover - executed when PyYAML missing
    yaml = None  # type: ignore[assignment]


def _convert_scalar(token: str) -> Any:
    lowered = token.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in token:
            return float(token)
        return int(token)
    except ValueError:
        return token


def loads_yaml(text: str) -> dict[str, Any]:
    if yaml is not None:
        return yaml.safe_load(text) or {}

    data: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            if current_key is None:
                raise ValueError(f"List item without key in line: {raw_line}")
            data.setdefault(current_key, [])
            assert isinstance(data[current_key], list)
            data[current_key].append(_convert_scalar(line[1:].strip()))
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not value:
                data[key] = []
                current_key = key
            else:
                data[key] = _convert_scalar(value)
                current_key = key if isinstance(data[key], list) else None
        else:
            raise ValueError(f"Unsupported config line: {raw_line}")
    return data


def load_yaml(path: str | Path) -> dict[str, Any]:
    return loads_yaml(Path(path).read_text(encoding="utf-8"))
