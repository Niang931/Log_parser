"""The structured form of a question. Whatever produced it (the deterministic fast path, the LLM,
or a person editing it before approval), it is validated here against what actually exists, and
anything it can't vouch for is refused (fail closed)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..enrich.validate import KEY_RE

INTENTS = ("general", "keyword", "event", "semantic", "rca", "summary")
LEVELS = ("debug", "info", "warn", "error")  # ascending
MAX_SPAN = timedelta(days=90)
DEFAULT_MINUTES = 60
MAX_KEYWORDS = 5
MAX_ROWS = 500
AGG_FNS = ("count", "count_distinct", "avg", "min", "max", "sum", "p95")
GROUP_FIELDS = ("host", "service", "source", "level", "status", "template_id")


class Aggregate(BaseModel):
    """A number instead of lines: ``fn`` of ``field`` per ``group_by`` (and per time bucket)."""
    model_config = ConfigDict(extra="forbid")
    fn: Literal["count", "count_distinct", "avg", "min", "max", "sum", "p95"] = "count"
    field: str | None = None
    group_by: list[str] = Field(default_factory=list, max_length=2)
    over_time: bool = False

    @field_validator("group_by")
    @classmethod
    def _groups(cls, v: list[str]) -> list[str]:
        for g in v:
            if g not in GROUP_FIELDS and not (g.startswith("@") and KEY_RE.match(g[1:])):
                raise ValueError(f"can't group by {g!r}; use {', '.join(GROUP_FIELDS)} or @extracted_field")
        return v

    @field_validator("field")
    @classmethod
    def _field(cls, v: str | None) -> str | None:
        if v is not None and v not in GROUP_FIELDS and v != "message" and not (v.startswith("@") and KEY_RE.match(v[1:])):
            raise ValueError(f"unknown field {v!r}")
        return v


class QuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Literal["general", "keyword", "event", "semantic", "rca", "summary"]
    minutes: int | None = Field(None, ge=1, le=60 * 24 * 90)  # relative: the last N minutes
    start: datetime | None = None  # absolute range; wins over minutes
    end: datetime | None = None
    services: list[str] = Field(default_factory=list, max_length=20)
    hosts: list[str] = Field(default_factory=list, max_length=20)
    min_level: Literal["debug", "info", "warn", "error"] | None = None
    keywords: list[str] = Field(default_factory=list, max_length=MAX_KEYWORDS)
    template_ids: list[int] = Field(default_factory=list, max_length=50)
    anchor_event: str | None = Field(None, max_length=200)  # "the server went down"
    anchor_time: datetime | None = None  # when the user says it happened
    semantic_query: str | None = Field(None, max_length=200)
    aggregate: Aggregate | None = None
    limit: int = Field(200, ge=1, le=MAX_ROWS)
    answer: str | None = Field(None, max_length=2000)  # general tier only: not from the logs

    @field_validator("keywords")
    @classmethod
    def _keywords(cls, v: list[str]) -> list[str]:
        out = [k.strip() for k in v if k.strip()]
        if any(len(k) > 100 for k in out):
            raise ValueError("a keyword is at most 100 characters")
        return list(dict.fromkeys(out))

    @field_validator("start", "end", "anchor_time")
    @classmethod
    def _aware(cls, v: datetime | None) -> datetime | None:
        return v if v is None or v.tzinfo else v.replace(tzinfo=UTC)


@dataclass
class Catalog:
    """What a spec may name: services, hosts and templates that exist."""
    services: set[str] = field(default_factory=set)
    hosts: set[str] = field(default_factory=set)
    templates: dict[int, str] = field(default_factory=dict)  # id -> text

    def describe(self, max_templates: int = 60) -> str:
        shown = list(self.templates.items())[:max_templates]
        return "\n".join([
            f"services: {', '.join(sorted(self.services)) or '(none)'}",
            f"hosts: {', '.join(sorted(self.hosts)) or '(none)'}",
            "templates (id: text):", *(f"  {i}: {t[:120]}" for i, t in shown),
            *([f"  ... and {len(self.templates) - len(shown)} more"] if len(self.templates) > len(shown) else [])])


def _pick(names: list[str], known: set[str], what: str) -> list[str]:
    """Exact names, matched without regard to case; an unknown one is refused, never guessed."""
    lower = {k.lower(): k for k in known}
    out = []
    for n in names:
        hit = lower.get(n.strip().lower())
        if hit is None:
            raise ValueError(f"unknown {what} {n!r}; known: {', '.join(sorted(known)) or 'none yet'}")
        out.append(hit)
    return list(dict.fromkeys(out))


def validate(raw: dict | QuerySpec, catalog: Catalog, now: datetime | None = None) -> QuerySpec:
    """A spec that is safe to plan from, or ValueError saying what's wrong."""
    now = now or datetime.now(UTC)
    try:
        spec = raw if isinstance(raw, QuerySpec) else QuerySpec.model_validate(raw)
    except ValidationError as error:
        first = error.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "spec"
        raise ValueError(f"invalid query spec ({where}): {first['msg']}") from None
    spec = spec.model_copy(deep=True)
    spec.services = _pick(spec.services, catalog.services, "service")
    spec.hosts = _pick(spec.hosts, catalog.hosts, "host")
    if unknown := [t for t in spec.template_ids if t not in catalog.templates]:
        raise ValueError(f"unknown template id {unknown[0]}")
    if spec.start or spec.end:
        end = min(spec.end or now, now + timedelta(minutes=1))  # nothing is logged in the future
        start = spec.start or end - timedelta(minutes=DEFAULT_MINUTES)
        if start >= end:
            raise ValueError("the time range starts after it ends")
        spec.start, spec.end = max(start, end - MAX_SPAN), end  # clamped to 90 days
        spec.minutes = None
    if spec.anchor_time and spec.anchor_time > now + timedelta(minutes=1):
        raise ValueError("the anchor time is in the future")
    if spec.intent == "semantic" and not (spec.semantic_query or spec.anchor_event or spec.keywords):
        raise ValueError("a semantic search needs something to search for")
    if spec.intent == "event" and not (spec.template_ids or spec.anchor_event or spec.keywords):
        raise ValueError("an event search needs a template, an event description or keywords")
    if spec.aggregate and spec.aggregate.fn != "count" and not spec.aggregate.field:
        raise ValueError(f"{spec.aggregate.fn} needs a field")
    if spec.intent != "general":
        spec.answer = None
    return spec


def time_range(spec: QuerySpec, now: datetime | None = None) -> tuple[datetime, datetime]:
    """The absolute range a spec covers. Relative ranges resolve against ``now`` (at approval,
    not when the question was asked)."""
    now = now or datetime.now(UTC)
    if spec.start and spec.end:
        return spec.start, spec.end
    if spec.anchor_time and not spec.minutes:
        return spec.anchor_time - timedelta(minutes=30), min(spec.anchor_time + timedelta(minutes=2), now)
    return now - timedelta(minutes=spec.minutes or DEFAULT_MINUTES), now


def levels_at_least(level: str | None) -> list[str]:
    return list(LEVELS[LEVELS.index(level):]) if level else []
