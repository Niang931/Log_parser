from __future__ import annotations

import re
from collections.abc import Sequence

SLOT_RE = re.compile(r"<\*>|<VAR:([A-Z0-9]+)>")
_KEY_PREFIX_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.\-]*)[=:]$")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_GENERIC_KEY_RE = re.compile(r"^var(_\d+)?$")
_INLINE_KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.\-]*)=(.*)$", re.DOTALL)

def slot_labels(tokens: Sequence[str]) -> list[str | None]:
    """Return one entry per variable slot: the mask label, or ``None`` for ``<*>``."""
    return [m.group(1) for tok in tokens for m in SLOT_RE.finditer(tok)]


def _snake(name: str) -> str:
    name = _CAMEL_RE.sub("_", name).replace("-", "_").replace(".", "_")
    return name.lower().strip("_") or "var"


def default_keys(tokens: Sequence[str]) -> list[str]:
    """Heuristic key names for each slot, used when no LLM names are available.

    ``temp=<VAR:NUM>`` yields ``temp``; a bare typed placeholder yields its
    label (``<VAR:IPV4>`` gives ``ipv4``); anything else falls back to ``var``.
    Duplicates get numeric suffixes so keys stay unique within a template.
    """
    keys: list[str] = []
    for tok in tokens:
        # The pattern is executed once before stopping so gotta move the start index to latest
        # to update the extractor to re-run again until all values are extracted
        for idx, match in enumerate(SLOT_RE.finditer(tok)):
            prefix = _KEY_PREFIX_RE.match(tok[: match.start()]) if idx == 0 else None
            if prefix:
                keys.append(_snake(prefix.group(1)))
            # if the variable name itself is the key. e.g: { name } then use that
            elif match.group(1):
                keys.append(match.group(1).lower())
            else:
                keys.append("var")

    # For each of the replicated keys, append the _<num> suffix after each count
    seen: dict[str, int] = {}
    unique: list[str] = []
    for key in keys:
        count = seen.get(key, 0)
        seen[key] = count + 1
        unique.append(key if count == 0 else f"{key}_{count}")
    return unique


def compile_extractor(tokens: Sequence[str]) -> re.Pattern[str]:
    parts: list[str] = []
    for tok in tokens:
        piece: list[str] = []
        last = 0
        for match in SLOT_RE.finditer(tok):
            piece.append(re.escape(tok[last : match.start()]))
            piece.append(r"(.+?)" if match.group(1) else r"(\S+?)")
            last = match.end()
        piece.append(re.escape(tok[last:]))
        parts.append("".join(piece))
    return re.compile(r"^\s*" + r"\s+".join(parts) + r"\s*$", re.DOTALL)


def extract(pattern: re.Pattern[str], keys: Sequence[str], line: str) -> dict[str, str] | None:
    """Return ``{key: value}`` for ``line``, or ``None`` if it does not fit the template."""
    match = pattern.match(line)
    if match is None:
        return None
    payload: dict[str, str] = {}
    for key, value in zip(keys, match.groups()):
        # Assume there are 2 logs user=A, user=B -> log parser would convert them to just <*>,
        # this is undesirable so we would like to recover the key as user=<*> instead
        inline = _INLINE_KV_RE.match(value) if _GENERIC_KEY_RE.match(key) else None
        if inline and _snake(inline.group(1)) not in payload:
            key, value = _snake(inline.group(1)), inline.group(2)
        payload[key] = value
    return payload
