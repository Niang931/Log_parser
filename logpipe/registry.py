"""Schema Registry: the single owner of template ids.

Drain's own cluster ids are renumbered whenever a tree is rebuilt, so
every id that leaves the parser (stored rows, LogBERT vocabulary) comes
from here instead.  Ids are allocated monotonically and never reused.
Each publish produces a new immutable :class:`RegistrySnapshot` with a
higher version; readers hold a snapshot and swap to the next one.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from deepparse.masks_types import Mask

from .parsing.kv import slot_labels


class UniqueTemplateID(Enum, int):
    UNSET_TEMPLATE_ID = 0
    UNK_TEMPLATE_ID = 1
    FIRST_TEMPLATE_ID = 2


@dataclass(frozen=True)
class TemplateEntry:
    template_id: int
    tokens: tuple[str, ...]
    keys: tuple[str, ...]
    alias_of: int | None = None


@dataclass(frozen=True)
class NewTemplate:
    """A template to be registered; ``alias_of`` points at an existing id."""

    tokens: tuple[str, ...]
    keys: tuple[str, ...]
    alias_of: int | None = None


@dataclass(frozen=True)
class RegistrySnapshot:
    version: int = 0
    next_id: int = UniqueTemplateID.FIRST_TEMPLATE_ID
    masks: tuple[Mask, ...] = ()
    templates: Mapping[int, TemplateEntry] = field(default_factory=lambda: MappingProxyType({}))
    key_vocab: frozenset[str] = frozenset()

    def canonical_id(self, template_id: int) -> int:
        entry = self.templates.get(template_id)
        while entry is not None and entry.alias_of is not None:
            template_id = entry.alias_of
            entry = self.templates.get(template_id)
        return template_id

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "next_id": self.next_id,
            "masks": [m.to_dict() for m in self.masks],
            "key_vocab": sorted(self.key_vocab),
            "templates": [
                {
                    "template_id": e.template_id,
                    "tokens": list(e.tokens),
                    "keys": list(e.keys),
                    "alias_of": e.alias_of,
                }
                for e in sorted(self.templates.values(), key=lambda e: e.template_id)
            ],
        }

    @classmethod
    def from_json(cls, payload: dict) -> RegistrySnapshot:
        templates = {
            t["template_id"]: TemplateEntry(
                template_id=t["template_id"],
                tokens=tuple(t["tokens"]),
                keys=tuple(t["keys"]),
                alias_of=t.get("alias_of"),
            )
            for t in payload["templates"]
        }
        return cls(
            version=payload["version"],
            next_id=payload["next_id"],
            masks=tuple(Mask.from_dict(m) for m in payload["masks"]),
            templates=MappingProxyType(templates),
            key_vocab=frozenset(payload.get("key_vocab", ())),
        )


class SchemaRegistry:
    def __init__(self, snapshot: RegistrySnapshot | None = None):
        self._snapshot = snapshot or RegistrySnapshot()
        self._lock = threading.Lock()

    @property
    def current(self) -> RegistrySnapshot:
        return self._snapshot

    def publish(
        self,
        new_templates: Sequence[NewTemplate] = (),
        *,
        masks: Iterable[Mask] | None = None,
        key_vocab: Iterable[str] = (),
    ) -> tuple[RegistrySnapshot, list[int]]:
        """Register templates and return the new snapshot plus their assigned ids."""
        with self._lock:
            base = self._snapshot
            templates = dict(base.templates)
            next_id = base.next_id
            assigned: list[int] = []
            for new in new_templates:
                if len(new.keys) != len(slot_labels(new.tokens)):
                    raise ValueError(
                        f"template {' '.join(new.tokens)!r} has "
                        f"{len(slot_labels(new.tokens))} slots but {len(new.keys)} keys"
                    )
                if new.alias_of is not None and new.alias_of not in templates:
                    raise ValueError(f"alias target {new.alias_of} is not registered")
                templates[next_id] = TemplateEntry(next_id, new.tokens, new.keys, new.alias_of)
                assigned.append(next_id)
                next_id += 1
            snapshot = RegistrySnapshot(
                version=base.version + 1,
                next_id=next_id,
                masks=tuple(masks) if masks is not None else base.masks,
                templates=MappingProxyType(templates),
                key_vocab=base.key_vocab | frozenset(key_vocab),
            )
            self._snapshot = snapshot
            return snapshot, assigned

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._snapshot.to_json(), indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> SchemaRegistry:
        return cls(RegistrySnapshot.from_json(json.loads(path.read_text(encoding="utf-8"))))
