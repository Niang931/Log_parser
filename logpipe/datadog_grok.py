"""logpipe parsing rules -> one Datadog log pipeline: a grok parser and a date remapper.

Each template comes with the newest line it parsed (``sample``: its extracted fields), so a
timestamp gets the exact ``date()`` format and a Datadog filter is only added when the value
really is what the field name says. Matching follows logpipe: whitespace is any run of it."""

from __future__ import annotations

import json
import re

from .parsing.grok import _PART

PIPELINE_NAME = "logpipe (auto)"

_MATCHER = {"notspace": "notSpace", "quoted": "quotedString", "path": "notSpace"}
_TIMESTAMP_RE = 'regex("[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}[^ ]*")'
_SPECIAL = re.compile(r"([\\^$.|?*+()\[\]{}])")
_SPACE = re.compile(r"\s+")
_DATE = re.compile(
    r"(\d{4}-\d{2}-\d{2})([T ])(\d{2}:\d{2}:\d{2})(?:([.,])(\d+))?(Z|[+-]\d{2}:\d{2}|[+-]\d{4})?"
)


def _is_json(v: str) -> bool:
    try:
        return isinstance(json.loads(v), dict | list)
    except ValueError:
        return False


# field name -> Datadog filter, and the check its sample value must pass
_FILTERS = (
    (r"ua|user_?agent|agent", "useragent", lambda v: bool(re.search(r"[A-Za-z]+/\d", v))),
    (r"url|uri|href", "url", lambda v: v.startswith(("http://", "https://"))),
    (
        r"query|query_?string|qs|params",
        "querystring",
        lambda v: bool(re.fullmatch(r"\??[^\s=&]+=[^\s]*", v)),
    ),
    (r"json|payload|body", "json", _is_json),
    (r"kv|pairs|attrs|attributes", "keyvalue", lambda v: bool(re.search(r"\w=\S", v))),
)


def date_pattern(value: str) -> str | None:
    """The ``date()`` arguments for a timestamp logpipe's ``timestamp`` matcher took, or None."""
    m = _DATE.fullmatch(value.strip())
    if m is None:
        return None
    _, sep, _, frac_sep, frac, zone = m.groups()
    pattern = "yyyy-MM-dd" + ("'T'" if sep == "T" else " ") + "HH:mm:ss"
    if frac:
        pattern += frac_sep + "S" * len(frac)
    if zone == "Z":
        return f'"{pattern}\'Z\'", "UTC"'
    if zone:
        pattern += "ZZ" if ":" in zone else "Z"
    return f'"{pattern}"'


def _filter(name: str, value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip().strip("\"'")
    for names, flt, ok in _FILTERS:
        if re.fullmatch(rf"(?:.*_)?(?:{names})", name) and ok(value):
            return flt
    return ""


def _literal(text: str) -> str:
    return r"\s+".join(_SPECIAL.sub(r"\\\1", part) for part in _SPACE.split(text))


def to_dd_rule(rule: str, sample: dict | None = None) -> tuple[str, list[str]]:
    """The rule in Datadog grok, and its fields parsed as dates (for the date remapper)."""
    sample = sample or {}
    out, dates, at = [], [], 0
    for m in _PART.finditer(rule):
        out.append(_literal(rule[at : m.start()]))
        matcher, arg, name = m.groups()
        flt = ""
        if arg is not None:
            part = f'regex("{arg}")'
        elif matcher == "timestamp":
            fmt = date_pattern(str(sample[name])) if name and sample.get(name) else None
            part = f"date({fmt})" if fmt else _TIMESTAMP_RE
            if fmt:
                dates.append(name)
        else:
            part = _MATCHER.get(matcher, matcher)
            flt = _filter(name, sample.get(name)) if name else ""
        out.append(f"%{{{part}{':' + name if name else ''}{':' + flt if flt else ''}}}")
        at = m.end()
    out.append(_literal(rule[at:]))
    return "".join(out), dates


def _specificity(rule: str) -> tuple[int, int]:
    """Datadog takes the first rule that matches: most fixed text first, fewest ``data`` next."""
    fixed = len(_SPACE.sub("", _PART.sub("", rule)))
    return -fixed, rule.count("%{data")


def _service_query(services: set[str]) -> str:
    if not services:
        return "*"
    quoted = (s if re.fullmatch(r"[\w.-]+", s) else json.dumps(s) for s in sorted(services))
    return f"service:({' OR '.join(quoted)})"


def pipeline(templates: list[dict]) -> dict:
    """``{"template_id", "rule", "sample": {field: value} | None, "service": str | None}`` ->
    the pipeline body, limited to the services the templates came from."""
    rules, dates, services = [], [], set()
    for t in sorted((t for t in templates if t.get("rule")), key=lambda t: _specificity(t["rule"])):
        text, date_fields = to_dd_rule(t["rule"], t.get("sample"))
        rules.append(f"t{t['template_id']} {text}")
        dates += [d for d in date_fields if d not in dates]
        if t.get("service"):
            services.add(t["service"])
    processors: list[dict] = [
        {
            "type": "grok-parser",
            "name": "logpipe templates",
            "is_enabled": True,
            "source": "message",
            "samples": [],
            "grok": {"support_rules": "", "match_rules": "\n".join(rules)},
        }
    ]
    if dates:
        processors.append(
            {
                "type": "date-remapper",
                "name": "logpipe timestamp",
                "is_enabled": True,
                "sources": dates,
            }
        )
    return {
        "name": PIPELINE_NAME,
        "is_enabled": True,
        "filter": {"query": _service_query(services)},
        "processors": processors,
    }
