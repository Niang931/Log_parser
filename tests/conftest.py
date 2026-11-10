from pathlib import Path

import pytest

from logpipe.calibrate import calibrate
from logpipe.enrich.llm import OfflineAdapter
from logpipe.settings import get_settings
from logpipe.sources.file_replay import replay

FIXTURES = Path(__file__).parent / "fixtures" / "iot"


@pytest.fixture(scope="session")
def fixture_lines() -> list[str]:
    return [event.line for event in replay(FIXTURES)]


@pytest.fixture(scope="session")
def snapshot(fixture_lines):
    return calibrate(fixture_lines, OfflineAdapter()).snapshot


@pytest.fixture(autouse=True)
def _fresh_settings():
    """Settings are read from the environment once per process; a test that changes the
    environment (monkeypatch.setenv) gets them read again."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
