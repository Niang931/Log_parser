"""Template review: templates mined from a sample wait here as drafts until a person approves,
edits, merges or rejects them. Only approved drafts reach the parser."""
from __future__ import annotations

from collections.abc import Sequence

from deepparse.masks_types import Mask

from ..enrich.validate import KEY_RE, template_problem
from ..parsing.frozen_parser import FrozenParser, Status
from ..parsing.grok import compile_rule, rule_from_template
from ..parsing.kv import SLOT_RE, slot_labels
from ..registry import FIRST_TEMPLATE_ID, NewTemplate, RegistrySnapshot, SchemaRegistry
from ..settings import get_settings


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
            if len(samples[i]) < get_settings().server.review_samples:
                samples[i].append(line)
    return [{"id": f"{batch_id}-{i}", "batch": batch_id, "tokens": list(t.tokens),
             "keys": list(t.keys), "count": counts[i], "samples": samples[i], "status": "pending",
             "rule": rule_from_template(t.tokens, t.keys), "rule_edited": False,
             "mined": {"tokens": list(t.tokens), "keys": list(t.keys)}}
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
    for line in lines[:get_settings().server.review_check_lines]:
        if parser.parse(line).status == Status.FINAL:
            matched += 1
            if len(samples) < get_settings().server.review_samples:
                samples.append(line)
    return {"count": matched, "samples": samples, "checked": min(len(lines), get_settings().server.review_check_lines)}


def merge_tokens(a: Sequence[str], b: Sequence[str]) -> list[str]:
    """One template covering both: positions that differ become variables."""
    if len(a) != len(b):
        raise ValueError(f"only templates with the same number of words can be merged "
                         f"({len(a)} vs {len(b)}); edit one first")
    return [x if x == y else x if SLOT_RE.fullmatch(x) else y if SLOT_RE.fullmatch(y) else "<*>"
            for x, y in zip(a, b)]


def check_rule(text: str, lines: Sequence[str], own=None) -> dict:
    """Run a parsing rule over sample lines: how many it matches, what it extracts from the first
    of them, and lines it misses that ``own`` (a predicate: this template's lines) says it should
    have matched."""
    rule = compile_rule(text)
    matched, rows, misses = 0, [], []
    for line in lines[:get_settings().server.review_check_lines]:
        fields = rule.extract(line)
        if fields is not None:
            matched += 1
            if len(rows) < 20:
                rows.append({"line": line, "fields": fields})
        elif own is not None and len(misses) < 10 and own(line):
            misses.append({"line": line, "fields": None})
    return {"count": matched, "checked": min(len(lines), get_settings().server.review_check_lines), "keys": list(rule.keys),
            "tokens": list(rule.tokens), "rows": rows + misses, "missed": len(misses),
            "samples": [r["line"] for r in rows[:get_settings().server.review_samples]]}


def mined_matcher(draft: dict, masks: Sequence[Mask]):
    """Whether a line belongs to the template as it was mined (before any edit)."""
    mined = draft.get("mined") or {"tokens": draft["tokens"], "keys": draft["keys"]}
    try:
        parser = _parser([NewTemplate(tuple(mined["tokens"]), tuple(mined["keys"]))], masks)
    except ValueError:
        return None
    return lambda line: parser.parse(line).status == Status.FINAL
