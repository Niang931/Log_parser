"""The two shapes a plan is made of: ``Filters`` (what a read selects, in the Query builder's
filter form, which every store's compiler checks and quotes) and ``Query`` (one read as shown for
approval, with the exact text that will run). Reads are compiled by the store adapter
(``logpipe.server.stores``) and run on its read-only path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from .spec import QuerySpec, levels_at_least


@dataclass
class Filters:
    services: list[str] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    template_ids: list[int] = field(default_factory=list)

    @classmethod
    def of(cls, spec: QuerySpec, groups: dict[str, list[str]] | None = None) -> Filters:
        """The spec's filters; a group named in it stands for the services of its streams."""
        services = list(
            dict.fromkeys(
                [*spec.services, *(s for g in spec.groups for s in (groups or {}).get(g, []))]
            )
        )
        return cls(
            services,
            list(spec.hosts),
            levels_at_least(spec.min_level),
            list(spec.keywords),
            list(spec.template_ids),
        )

    def builder(self) -> list[dict]:
        """As the Query builder's filter list (the builders quote and check every value)."""
        out = []
        for name, values in (
            ("service", self.services),
            ("host", self.hosts),
            ("level", self.levels),
        ):
            if values:
                out.append(
                    {
                        "field": name,
                        "op": "in",
                        "value": ",".join(v.replace(",", " ") for v in values),
                    }
                )
        if self.template_ids:
            out.append(
                {
                    "field": "template_id",
                    "op": "in",
                    "value": ",".join(str(int(t)) for t in self.template_ids),
                }
            )
        out += [{"field": "message", "op": "contains", "value": k} for k in self.keywords]
        return out

    def describe(self) -> str:
        parts = [f"{k}: {', '.join(map(str, v))}" for k, v in asdict(self).items() if v]
        return "; ".join(parts) or "no filters"


# reads compiled for the log store, and reads of the server's own state
LogTool = Literal["rows", "template_counts", "buckets", "aggregate"]
LocalTool = Literal["agents", "alerts"]
Tool = LogTool | LocalTool


@dataclass
class Query:
    """One read, as shown for approval: what it is for, and the exact text that will run."""

    id: str
    tool: Tool
    title: str
    why: str
    args: dict
    compiled: str
    engine: str  # the store's engine, or "local" for reads that touch no log data
    status: str = "pending"  # pending | approved | rejected | done | failed
    error: str | None = None
    result_summary: str | None = None
