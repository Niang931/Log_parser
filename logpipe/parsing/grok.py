"""Datadog-style parsing rules: ``user %{integer:user_id} logged in from %{ip:client_ip}``.

A rule is literal text with ``%{matcher:field}`` parts. Whitespace in the rule matches any run
of whitespace; ``%{matcher}`` without a name matches without extracting;
``%{regex("[a-f0-9]{8}"):id}`` takes a custom regex. A rule must match the whole line.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..enrich.validate import KEY_RE
from .kv import SLOT_RE

MATCHERS = {
    "word": r"\w+",
    "notspace": r"\S+",
    "integer": r"[-+]?\d+",
    "number": r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
    "ip": r"(?:\d{1,3}(?:\.\d{1,3}){3}|[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7})",
    "hostname": r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?",
    "path": r"(?:/[^\s?#]*)+",
    "uuid": r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}",
    "quoted": r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'",
    "timestamp": r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?",
    "data": r".*?",
}
_PART = re.compile(r'%\{(\w+)(?:\("((?:[^"\\]|\\.)*)"\))?(?::([^}]*))?\}')
# Nested quantifiers such as (a+)+ backtrack catastrophically on long lines.
_NESTED = re.compile(r"\([^()]*[+*][^()]*\)[+*{]")
# A mask label from calibration -> the matcher a default rule uses for it.
_LABEL_MATCHER = (("TIME", "data"), ("DATE", "data"), ("IP", "ip"), ("UUID", "uuid"),
                  ("PATH", "path"), ("URL", "notspace"), ("INT", "integer"), ("NUM", "number"))


@dataclass(frozen=True)
class Rule:
    text: str
    regex: re.Pattern[str]
    keys: tuple[str, ...]
    tokens: tuple[str, ...]  # the rule as a Drain template: each %{...} becomes <*>

    def extract(self, line: str) -> dict[str, str] | None:
        m = self.regex.match(line)
        return None if m is None else {k: v for k, v in m.groupdict().items() if v is not None}


def compile_rule(text: str) -> Rule:
    """Raises ValueError with a message meant for the person who wrote the rule."""
    text = text.strip()
    if not text:
        raise ValueError("the rule is empty")
    pattern, keys, shape, pos = [], [], [], 0
    for m in _PART.finditer(text):
        literal = text[pos:m.start()]
        pattern.append(_literal(literal))
        shape.append(literal)
        matcher, arg, name = m.group(1), m.group(2), m.group(3)
        if matcher == "regex":
            if arg is None:
                raise ValueError('regex needs a pattern: %{regex("[0-9a-f]+"):name}')
            body = arg.replace('\\"', '"')
            if _NESTED.search(body):
                raise ValueError(f"regex {body!r} nests quantifiers like (a+)+, which can hang on long lines")
            try:
                re.compile(body)
            except re.error as error:
                raise ValueError(f"regex {body!r}: {error}") from error
        elif matcher not in MATCHERS:
            raise ValueError(f"unknown matcher %{{{matcher}}}; use one of {', '.join([*MATCHERS, 'regex'])}")
        else:
            body = MATCHERS[matcher]
        if name:
            if not KEY_RE.match(name):
                raise ValueError(f"field name {name!r} must be lower_snake_case")
            if name in keys:
                raise ValueError(f"field {name!r} is used twice")
            keys.append(name)
            pattern.append(f"(?P<{name}>{body})")
        else:
            pattern.append(f"(?:{body})")
        shape.append("<*>")
        pos = m.end()
    pattern.append(_literal(text[pos:]))
    shape.append(text[pos:])
    if any("%{" in lit for lit in shape if lit != "<*>"):
        raise ValueError("a %{...} part isn't closed or is malformed, e.g. %{integer:status}")
    tokens = tuple("".join(shape).split())
    if not any(not SLOT_RE.fullmatch(t) for t in tokens):
        raise ValueError("the rule needs at least one fixed word, or it would match every line")
    try:
        regex = re.compile(r"^\s*" + "".join(pattern) + r"\s*$", re.DOTALL)
    except re.error as error:
        raise ValueError(f"the rule doesn't compile: {error}") from error
    return Rule(text, regex, tuple(keys), tokens)


def _literal(text: str) -> str:
    return r"\s+".join(re.escape(part) for part in re.split(r"\s+", text)) if text else ""


def rule_from_template(tokens: Sequence[str], keys: Sequence[str]) -> str:
    """A starting rule for a mined template: its words, with each slot as a typed matcher."""
    k = iter(keys)
    out = []
    for tok in tokens:
        def slot(m: re.Match) -> str:
            label = (m.group(1) or "").upper()
            matcher = next((mt for frag, mt in _LABEL_MATCHER if frag in label), "notspace")
            return f"%{{{matcher}:{next(k, 'field')}}}"
        out.append(SLOT_RE.sub(slot, tok))
    return " ".join(out)
