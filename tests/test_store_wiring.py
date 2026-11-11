"""`--store mongodb` is wired into shared code (no MongoDB needed)."""

from __future__ import annotations

import pytest

from logpipe.cli import main
from logpipe.server.stores import reads_for


@pytest.mark.parametrize("cmd", ["run", "worker", "demo", "server", "bench"])
def test_cli_accepts_mongodb(cmd):
    with pytest.raises(SystemExit) as ok:
        main([cmd, "--store", "mongodb", "--help"])  # help exits 0 only if the choice passed
    assert ok.value.code == 0


def test_cli_rejects_unknown_store():
    with pytest.raises(SystemExit) as bad:
        main(["server", "--store", "bogus", "--help"])
    assert bad.value.code == 2


def test_reads_for_unknown_store_raises():
    class Odd: ...

    with pytest.raises(ValueError, match="no reads for store Odd"):
        reads_for(Odd())
