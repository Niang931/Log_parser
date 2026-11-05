"""Template review: templates mined from a sample wait here as drafts until a person approves,
edits, merges or rejects them. Only approved drafts reach the parser."""
from __future__ import annotations

from collections.abc import Sequence

from deepparse.masks_types import Mask

from ..enrich.validate import KEY_RE, template_problem
from ..parsing.frozen_parser import FrozenParser, Status
from ..parsing.kv import SLOT_RE, slot_labels
from ..registry import FIRST_TEMPLATE_ID, NewTemplate, RegistrySnapshot, SchemaRegistry

SAMPLES = 5
# shortcut: an edit is re-checked against at most this many sample lines; enough for a live count.
CHECK_LINES = 20_000


def _parser(templates: Sequence[NewTemplate], masks: Sequence[Mask]) -> FrozenParser:
    snapshot, _ = SchemaRegistry(RegistrySnapshot(masks=tuple(masks))).publish(templates)
    return FrozenParser(snapshot)


def draft_docs(batch_id: str, templates: Sequence[NewTemplate], masks: Sequence[Mask],
               lines: Sequence[str]) -> list[dict]:
    """One draft per template, with how many sample lines it matches and a few of them."""
    parser = _parser(templates, masks)
    counts = [0] * len(templates)
    samples: list[list[str]] = [[] for _ in templates]
    for line in lines:
        result = parser.parse(line)
        if result.status == Status.FINAL:
            i = result.template_id - FIRST_TEMPLATE_ID
            counts[i] += 1
            if len(samples[i]) < SAMPLES:
                samples[i].append(line)
    return [{"id": f"{batch_id}-{i}", "batch": batch_id, "tokens": list(t.tokens),
             "keys": list(t.keys), "count": counts[i], "samples": samples[i], "status": "pending"}
            for i, t in enumerate(templates)]


def validate(tokens: Sequence[str], keys: Sequence[str]) -> None:
    """Reject an edited template the parser can't use; the message is shown under the editor."""
    if not tokens:
        raise ValueError("the template is empty")
    problem = template_problem(tokens, [])
    if problem:
        raise ValueError(problem)
    slots = len(slot_labels(tokens))
    if len(keys) != slots:
        raise ValueError(f"{slots} variable slots (<*>) but {len(keys)} field names")
    bad = [k for k in keys if not KEY_RE.match(k)]
    if bad:
        raise ValueError(f"field names must be lower_snake_case: {', '.join(bad)}")
    if len(set(keys)) != len(keys):
        raise ValueError("field names must be unique")


def check(tokens: Sequence[str], keys: Sequence[str], masks: Sequence[Mask],
          lines: Sequence[str]) -> dict:
    """How many sample lines this template alone matches, and a few of them."""
    validate(tokens, keys)
    parser = _parser([NewTemplate(tuple(tokens), tuple(keys))], masks)
    matched, samples = 0, []
    for line in lines[:CHECK_LINES]:
        if parser.parse(line).status == Status.FINAL:
            matched += 1
            if len(samples) < SAMPLES:
                samples.append(line)
    return {"count": matched, "samples": samples, "checked": min(len(lines), CHECK_LINES)}


def merge_tokens(a: Sequence[str], b: Sequence[str]) -> list[str]:
    """One template covering both: positions that differ become variables."""
    if len(a) != len(b):
        raise ValueError(f"only templates with the same number of words can be merged "
                         f"({len(a)} vs {len(b)}); edit one first")
    return [x if x == y else x if SLOT_RE.fullmatch(x) else y if SLOT_RE.fullmatch(y) else "<*>"
            for x, y in zip(a, b)]
