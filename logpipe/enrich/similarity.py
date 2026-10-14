"""Template similarity and alias decisions.

A newly mined template is compared against existing canonical templates
with a token-level, slot-aware similarity (Drain's measure, extended to
templates of different lengths).  Candidates above the threshold are then
classified by *how* they differ:

* ``slot_only``: they differ only where one side has a variable slot. This
  is the same event at a different generalisation, so alias automatically.
* ``wording`` / ``literal``: extra or replaced literal words. Ask the LLM
  adapter to confirm, unless the replaced words are opposite states
  (opened/closed, on/off, ...), which are never merged.

Embedding similarity is deliberately not used here: sentence embeddings
place "Door opened" and "Door closed" above 0.9 cosine, and merging those
erases exactly the distinction anomaly detection depends on.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from ..parsing.kv import SLOT_RE
from ..registry import RegistrySnapshot
from .llm import LLMAdapter
from .worker import GuardedAdapter

DiffKind = Literal["identical", "slot_only", "wording", "literal", "conflict"]

# TODO: this shit was just generated from AI, should consider finding a list online
_OPPOSITES = [
    ("open", "close"),
    ("opened", "closed"),
    ("on", "off"),
    ("up", "down"),
    ("start", "stop"),
    ("started", "stopped"),
    ("success", "failure"),
    ("succeeded", "failed"),
    ("success", "fail"),
    ("ok", "error"),
    ("pass", "fail"),
    ("passed", "failed"),
    ("connected", "disconnected"),
    ("connect", "disconnect"),
    ("enabled", "disabled"),
    ("enable", "disable"),
    ("locked", "unlocked"),
    ("lock", "unlock"),
    ("true", "false"),
    ("yes", "no"),
    ("high", "low"),
    ("accepted", "rejected"),
    ("accept", "reject"),
    ("granted", "denied"),
    ("online", "offline"),
    ("valid", "invalid"),
    ("armed", "disarmed"),
    ("added", "removed"),
    ("attached", "detached"),
    ("acquired", "released"),
    ("begin", "end"),
    ("enter", "exit"),
    ("entered", "exited"),
    ("in", "out"),
    ("min", "max"),
    ("increase", "decrease"),
    ("above", "below"),
    ("allow", "deny"),
]
OPPOSITES = {frozenset(pair) for pair in _OPPOSITES}
# Words that mark a failed or rejected outcome.  A template that has one of
# these where the other template has none describes a different outcome.
NEGATIVE_OUTCOMES = {
    "fail",
    "failed",
    "failure",
    "error",
    "errors",
    "rejected",
    "denied",
    "refused",
    "timeout",
    "timed",
    "invalid",
    "aborted",
    "abort",
    "crashed",
    "crash",
    "lost",
    "dropped",
    "unreachable",
    "unavailable",
    "fault",
    "panic",
    "corrupt",
    "corrupted",
}
_NEGATING_PREFIXES = ("un", "dis", "in", "non", "de")
_WORD_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class AliasDecision:
    alias_of: int | None
    score: float
    kind: DiffKind | None
    reason: str


def _is_slot(token: str) -> bool:
    return SLOT_RE.search(token) is not None


def _slot_regex(token: str) -> re.Pattern[str]:
    """Replaces the <*> placeholder with .+? and return compiled the expression."""
    parts, last = [], 0
    for match in SLOT_RE.finditer(token):
        parts.append(re.escape(token[last : match.start()]))
        parts.append(r".+?")
        last = match.end()
    parts.append(re.escape(token[last:]))
    return re.compile("".join(parts))


def tokens_compatible(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` can describe the same token (equal, or a slot covers the other)."""
    if a == b:
        return True
    if _is_slot(a) and _is_slot(b):
        return True
    if _is_slot(a):
        return _slot_regex(a).fullmatch(b) is not None
    if _is_slot(b):
        return _slot_regex(b).fullmatch(a) is not None
    return False


def _norm(token: str) -> str:
    return "<*>" if _is_slot(token) else token


def template_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """Slot-aware token similarity in [0, 1]."""
    if not a or not b:
        return 0.0
    # Resolve to this simple way of doing things if the number of tokens is the same
    if len(a) == len(b):
        return sum(tokens_compatible(x, y) for x, y in zip(a, b)) / len(a)
    return SequenceMatcher(None, [_norm(t) for t in a], [_norm(t) for t in b]).ratio()


def _words(tokens: Sequence[str]) -> set[str]:
    return {w for tok in tokens if not _is_slot(tok) for w in _WORD_RE.findall(tok.lower())}


def _forms(word: str) -> set[str]:
    """The word plus crude inflection-stripped forms: closed → {closed, close, clos}."""
    forms = {word}
    for suffix in ("ing", "ed", "d", "s"):
        # The 2 here is consider a word should have at least 2 chars
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            forms.add(word[: -len(suffix)])  # Remove the suffix
    return forms


def _opposed(x: str, y: str) -> bool:
    for fx in _forms(x):
        for fy in _forms(y):
            if frozenset((fx, fy)) in OPPOSITES:
                return True
            if any(fx == p + fy or fy == p + fx for p in _NEGATING_PREFIXES):
                return True
    return False


def opposite_states(a_tokens: Sequence[str], b_tokens: Sequence[str]) -> bool:
    """True when the words only one side uses include an opposite pair (open/closed, un-lock...)."""
    only_a = _words(a_tokens) - _words(b_tokens)
    only_b = _words(b_tokens) - _words(a_tokens)

    # The & here is AND operation for sets (not boolean)
    # If template a is negative and template b is positive then theres a mismatch
    # so we flag it immediately or we check for opposing using suf/pref-ixes later
    if bool(only_a & NEGATIVE_OUTCOMES) != bool(only_b & NEGATIVE_OUTCOMES):
        return True
    return any(_opposed(x, y) for x in only_a for y in only_b)


def classify_difference(a: Sequence[str], b: Sequence[str]) -> DiffKind:
    if list(a) == list(b):
        return "identical"
    if opposite_states(a, b):
        return "conflict"
    if len(a) == len(b):
        return "slot_only" if all(tokens_compatible(x, y) for x, y in zip(a, b)) else "literal"

    # different length comparison
    # NOTE: using python built-in sequence matcher which might record the following
    # opcodes: equal, delete, insert, replace, we only catch replace since the
    # original word was completely change
    matcher = SequenceMatcher(None, [_norm(t) for t in a], [_norm(t) for t in b])
    if any(op == "replace" for op, *_ in matcher.get_opcodes()):
        return "literal"
    return "wording"


def find_alias(
    tokens: Sequence[str],
    snapshot: RegistrySnapshot,
    adapter: LLMAdapter | GuardedAdapter,
    samples: Sequence[str],
    *,
    threshold: float = 0.75,
    max_candidates: int = 3,
) -> AliasDecision:
    """Pick the existing canonical template ``tokens`` should alias, if any."""
    scored = sorted(
        (
            (template_similarity(tokens, entry.tokens), entry)
            for entry in snapshot.templates.values()
            if entry.alias_of is None
        ),
        # sort the tuple by its similarity score then entry template id
        key=lambda pair: (-pair[0], pair[1].template_id),
    )
    for score, entry in scored[:max_candidates]:
        if score < threshold:
            break
        kind = classify_difference(tokens, entry.tokens)
        if kind in ("identical", "slot_only"):
            return AliasDecision(entry.template_id, score, kind, "differs only in variable slots")
        # TODO: currently I dont really handle the conflict state in the architecture (discuss later)
        if kind == "conflict":
            continue
        # ask LLM to output the final verdict
        if adapter.confirm_alias(tokens, entry.tokens, samples):
            return AliasDecision(entry.template_id, score, kind, "confirmed by LLM")
    best = scored[0][0] if scored else 0.0  # scored[0][0] is entry with highest similarity score
    return AliasDecision(None, best, None, "no compatible template above threshold")
