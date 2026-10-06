from pathlib import Path

import pytest

from logpipe.calibrate import calibrate
from logpipe.enrich.llm import OfflineAdapter
from logpipe.sources.file_replay import replay

FIXTURES = Path(__file__).parent / "fixtures" / "iot"


@pytest.fixture(scope="session")
def fixture_lines() -> list[str]:
    return [event.line for event in replay(FIXTURES)]


@pytest.fixture(scope="session")
def snapshot(fixture_lines):
    return calibrate(fixture_lines, OfflineAdapter())
