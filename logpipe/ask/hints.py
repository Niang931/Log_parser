"""Value hints: which host, source or extracted value the trouble concentrates on.

Templates say *what* went wrong; a hint says *where*: "host=web-3 is on 96% of the error lines and
2% of the others". For every value of every field, compare its share among the error/warn lines of
the window with its share among the other lines there; a value far more common among errors is a
hint (the contrastive skew of Log-Insight, arXiv 2607.08529). Fields that look like ids (most
values distinct) or that only the error lines carry (an error template's own slot) are skipped:
they would always "concentrate" and say nothing.
"""
from __future__ import annotations

from collections import Counter

from ..settings import RcaSettings
from .models import Hint, LogLine

COLUMNS = ("host", "source")


def _values(line: LogLine) -> dict[str, str]:
    out = {c: getattr(line, c) for c in COLUMNS if getattr(line, c)}
    for k, v in line.payload.items():
        if v is not None and not isinstance(v, (dict, list)) and str(v).strip():
            out["@" + k] = str(v).strip()[:80]
    return out


def value_hints(errors: list[LogLine], others: list[LogLine], cfg: RcaSettings) -> list[Hint]:
    if not errors or len(others) < cfg.hint_min_others:
        return []
    err, oth = [_values(r) for r in errors], [_values(r) for r in others]
    fields = {f for v in err for f in v}
    found = []
    for f in fields:
        e = Counter(v[f] for v in err if f in v)
        o = Counter(v[f] for v in oth if f in v)
        carried = sum(o.values())
        if carried < cfg.hint_min_lines:  # only the error lines carry it: the error template's own slot
            continue
        distinct = set(e) | set(o)
        if len(distinct) > 50 or len(distinct) > 0.5 * (sum(e.values()) + carried):  # an id: every line its own value
            continue
        for value, n in e.items():
            share = n / len(err)
            other = (o[value] + 0.5) / (len(oth) + 1)  # smoothed: a value no other line has isn't infinitely skewed
            ratio = share / other
            if n >= cfg.hint_min_lines and share >= cfg.hint_min_share and ratio >= cfg.hint_min_ratio:
                found.append((share, ratio, f, value, n, o[value] / len(oth)))
    found.sort(key=lambda h: (-h[0], -h[1], h[2], h[3]))
    return [Hint(ref=f"H{i}", field=f, value=v, errors=n, error_share=round(share, 3), other_share=round(other, 3),
                 ratio=round(ratio, 1))
            for i, (share, ratio, f, v, n, other) in enumerate(found[:cfg.max_hints], 1)]
