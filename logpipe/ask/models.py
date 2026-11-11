"""The typed shapes Ask passes around: rows read from the store, the root-cause report (anchor,
ranked candidates, evidence lines, the answer) and every kind of result a plan ends with.

They are pydantic models so a plan can be stored as JSON (``model_dump(mode="json")``) and read
back (``model_validate``) with the same field names the UI reads.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field

from ..registry import UNK_TEMPLATE_ID


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _aware(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=UTC)


Utc = Annotated[datetime, AfterValidator(_aware)]  # a time without a zone is UTC
# JSON text or a document
Fields = Annotated[
    dict, BeforeValidator(lambda v: json.loads(v) if isinstance(v, str) else v or {})
]


def template_ref(tid: int | None) -> str | None:
    """``T12`` for a template; nothing for an unparsed line (they all share one id, so citing it says nothing)."""
    return None if tid is None or tid == UNK_TEMPLATE_ID else f"T{tid}"


# Rows the reads return (one model per compiled read)
class TemplateCount(_Model):
    """One (template, service) in a time range: ``template_counts``."""

    template_id: int
    service: str | None = None
    host: str | None = None
    n: int
    first: Utc
    last: Utc
    example: str = ""
    log_id: str | None = None
    level: str | None = None


class Bucket(_Model):
    """Lines of one key (a template id or a service) in one time bucket: ``buckets``."""

    key: str | int | None
    bucket: int
    n: int


class LogLine(_Model):
    """One line: ``rows``."""

    log_id: str | None = None
    timestamp: Utc
    host: str | None = None
    service: str | None = None
    source: str | None = None
    level: str | None = None
    template_id: int | None = None
    message: str
    # the template's extracted fields, when the read asked for them
    payload: Fields = Field(default_factory=dict)


class AgentSeen(_Model):
    host: str | None
    last_seen: Utc | None = None
    online: bool | None = None


class Alert(_Model):
    """A LogBERT alert window (``reads.alerts``); ``key`` is host/service."""

    key: str
    window_start: Utc
    window_end: Utc
    miss_ratio: float = 0.0


class TimeSpan(_Model):
    start: Utc
    end: Utc


# The root-cause report
AnchorKind = Literal["user", "silence", "shutdown", "logbert", "new_error", "range"]


class Anchor(_Model):
    """When things broke, and why that moment was picked."""

    kind: AnchorKind
    at: Utc
    why: str
    service: str | None = None
    host: str | None = None
    template_id: int | None = None
    message: str | None = None  # the line at the anchor, when there is one
    loud: bool = False  # silence: the service's last line was an error or a shutdown message


class Candidate(_Model):
    """A (template, service) seen in the window, with every number that decides its rank."""

    template_id: int
    service: str | None
    host: str | None = None
    count: int
    first: Utc
    last: Utc
    level: str | None = None
    example: str
    log_id: str | None = None
    template: str
    new: bool
    baseline_median: float
    anomaly: float
    rarity: float
    proximity: float
    score: float
    why: str
    earliest_in_chain: bool = False


class Evidence(_Model):
    """A line in the bundle the LLM sees, cited as ``ref`` (L1 is the earliest)."""

    ref: str
    timestamp: Utc
    service: str | None = None
    host: str | None = None
    level: str | None = None
    template_id: int | None = None
    log_id: str | None = None
    message: str
    why: str
    score: float | None = None


class Hint(_Model):
    """A value error lines carry far more often than the other lines in the window, cited as ``ref``
    (H1 is the strongest): the host, source or extracted field the trouble concentrates on."""

    ref: str
    field: str  # host, source, or an extracted field (@name)
    value: str
    errors: int  # error/warn lines carrying it
    error_share: float
    other_share: float
    ratio: float


class Series(_Model):
    """Error/warn (or all) lines of one service per time step, read by a follow-up, cited as ``ref`` (S1, ...)."""

    ref: str
    service: str | None
    what: str
    start: Utc
    step_seconds: int
    counts: list[int]


class TimelineEvent(_Model):
    at: str = ""
    what: str
    cites: list[str] = Field(default_factory=list)


class Hypothesis(_Model):
    """A likely cause (from the LLM) or a lead (the ranking itself, ``confidence="ranked"``)."""

    cause: str
    confidence: str = "low"
    reasoning: str = ""
    cites: list[str] = Field(default_factory=list)


class Dropped(_Model):
    """A claim removed because it cited nothing in the bundle."""

    section: Literal["timeline", "hypotheses"]
    text: str | None = None
    cites: list[str] = Field(default_factory=list)


class Answer(_Model):
    summary: str
    timeline: list[TimelineEvent] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    leads: list[Hypothesis] = Field(default_factory=list)
    insufficient_evidence: bool = False
    missing: str = ""
    next_steps: list[str] = Field(default_factory=list)
    dropped: list[Dropped] = Field(default_factory=list)
    by: Literal["llm", "rules"] = "rules"
    note: str = ""
    llm_error: str = ""  # the LLM failed or timed out: the answer is the rules' ranking instead


class Report(_Model):
    kind: Literal["report"] = "report"
    intent: str
    anchor: Anchor
    candidates: list[Candidate] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    # group -> its services, for the services in the bundle
    groups: dict[str, list[str]] = Field(default_factory=dict)
    hints: list[Hint] = Field(default_factory=list)
    series: list[Series] = Field(default_factory=list)
    answer: Answer | None = None

    def refs(self) -> set[str]:
        """Every ref a claim may cite: lines, value hints, rates, and the templates of lines and candidates."""
        out = (
            {e.ref for e in self.evidence}
            | {h.ref for h in self.hints}
            | {s.ref for s in self.series}
        )
        out |= {template_ref(e.template_id) for e in self.evidence}
        return (out | {template_ref(c.template_id) for c in self.candidates}) - {None}  # type: ignore[operator]


# The other ways a plan ends
class GeneralResult(_Model):
    kind: Literal["general"] = "general"
    answer: str
    note: str = "From general knowledge, not from your logs."


class ClarifyResult(_Model):
    kind: Literal["clarify"] = "clarify"
    message: str
    note: str = ""


class TemplateSeen(_Model):
    template_id: int
    service: str | None
    count: int
    first: Utc
    last: Utc
    text: str


class LinesResult(_Model):
    kind: Literal["lines"] = "lines"
    lines: list[LogLine]
    templates: list[TemplateSeen] = Field(default_factory=list)
    note: str = ""


class TableResult(_Model):
    kind: Literal["table"] = "table"
    columns: list[str]
    rows: list[list]
    compiled: str


Result = Annotated[
    GeneralResult | ClarifyResult | LinesResult | TableResult | Report, Field(discriminator="kind")
]


class RcaState(_Model):
    """Where a root-cause plan is between rounds (stored with the plan)."""

    phase: Literal["discover", "evidence", "wide", "follow_up", "clarify", "done"]
    range: TimeSpan | None = None
    anchor: Anchor | None = None
    report: Report | None = None
    follow_ups: int = 0
