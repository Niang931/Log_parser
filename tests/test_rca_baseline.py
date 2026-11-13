"""The RCA baseline: whole window-sized slices that end where the window starts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from logpipe.ask.engine import Ctx
from logpipe.ask.models import Anchor
from logpipe.ask.rca import Around, evidence_round, rank
from logpipe.ask.spec import QuerySpec
from logpipe.server.stores.memory import MemoryReads
from logpipe.settings import RcaSettings
from logpipe.storage.memory import MemoryStore

AT = datetime(2024, 11, 22, 14, 59, tzinfo=UTC)


def test_the_slice_just_before_the_window_counts():
    # a 13-minute window doesn't divide 24 h: the newest slice used to be cut off and dropped
    cfg = RcaSettings(before_minutes=5, after_minutes=8)
    ctx = Ctx(store=MemoryReads(MemoryStore()), templates={}, now=AT + timedelta(hours=1), rca=cfg)
    anchor = Anchor(kind="user", at=AT, why="given")
    spec = QuerySpec(intent="rca")
    done = evidence_round(2, anchor, ctx, spec)
    base = next(q for q in done.queries if q.tool == "buckets").args
    start, end = (datetime.fromisoformat(base[k]) for k in ("start", "end"))
    assert end == AT - cfg.before
    assert int((end - start).total_seconds()) % base["step"] == 0
    n = int((end - start).total_seconds()) // base["step"]

    # template 1 logged 4 lines in the slice just before the window, and 40 in the window
    count = {
        "template_id": 1,
        "service": "api",
        "n": 40,
        "first": AT.isoformat(),
        "last": AT.isoformat(),
        "level": "info",
    }
    outputs = {
        f"r2q{Around.TEMPLATES}": [count],
        f"r2q{Around.BASELINE}": [{"key": 1, "bucket": n - 1, "n": 4}],
    }
    [cand], _, _ = rank(spec, anchor, done, outputs, ctx)
    assert "no baseline" not in cand.why and not cand.new
    assert cand.rarity == 1.0  # 10x its baseline
