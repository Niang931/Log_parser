from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..registry import RegistrySnapshot, UniqueTemplateID
from .engine import make_engine
from .grok import compile_rule
from .kv import compile_extractor, extract


class Status(StrEnum):
    FINAL = "FINAL"
    PENDING = "PENDING"


@dataclass(frozen=True, slots=True)
class ParseResult:
    template_id: int
    status: Status
    payload: dict[str, str]
    registry_version: int


class FrozenParser:
    def __init__(self, snapshot: RegistrySnapshot, depth: int = 5):
        self.snapshot = snapshot
        self._engine = make_engine(snapshot.masks, depth=depth)
        self._extractors = {}
        self._rules = []
        for entry in snapshot.templates.values():
            self._engine.add_template(entry.template_id, entry.tokens)
            self._extractors[entry.template_id] = (compile_extractor(entry.tokens), entry.keys)
            if entry.rule:
                self._rules.append((entry.template_id, compile_rule(entry.rule)))

    def parse(self, line: str) -> ParseResult:
        version = self.snapshot.version
        # shortcut: hand-written rules are tried one by one before the tree; fine for the tens a
        # person writes, index them by first literal word if there are ever thousands.
        for template_id, rule in self._rules:
            payload = rule.extract(line)
            if payload is not None:
                return ParseResult(self.snapshot.canonical_id(template_id), Status.FINAL, payload, version)
        cluster = self._engine.match(line)

        # If there is no pre-defined cluster
        if cluster is None:
            return ParseResult(UniqueTemplateID.UNK_TEMPLATE_ID, Status.PENDING, {}, version)

        pattern, keys = self._extractors[cluster.cluster_id]
        payload = extract(pattern, keys, line) or {}
        return ParseResult(
            self.snapshot.canonical_id(cluster.cluster_id), Status.FINAL, payload, version
        )


class ParserHandle:
    def __init__(self, snapshot: RegistrySnapshot):
        self._parser = FrozenParser(snapshot)

    @property
    def version(self) -> int:
        return self._parser.snapshot.version

    def parse(self, line: str) -> ParseResult:
        return self._parser.parse(line)

    def swap(self, snapshot: RegistrySnapshot) -> None:
        if snapshot.version <= self.version:
            return
        self._parser = FrozenParser(snapshot)
