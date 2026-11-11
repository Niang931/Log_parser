"""Settings come from the environment through one place, and UI settings are validated."""

from __future__ import annotations

import pytest

from logpipe.settings import RuntimeSettings, get_settings


def test_each_group_reads_its_prefix(monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_PORT", "9123")
    monkeypatch.setenv("LOGPIPE_RCA_BEFORE_MINUTES", "45")
    monkeypatch.setenv("WEB_BROWSER", "chromium")
    get_settings.cache_clear()
    s = get_settings()
    assert (
        s.clickhouse.port == 9123
        and s.rca.before.total_seconds() == 45 * 60
        and s.llm.web_browser == "chrome"
    )
    with pytest.raises(Exception):  # noqa: B017  frozen
        s.rca.before_minutes = 1  # type: ignore[misc]


def test_runtime_settings_change_only_editable_fields_within_bounds():
    rs = RuntimeSettings()
    changed = rs.changed({"sample_cap": 1000, "calibrated": True, "ask_require_approval": False})
    assert (
        changed.sample_cap == 1000 and not changed.calibrated and not changed.ask_require_approval
    )
    with pytest.raises(ValueError, match="sample_cap"):
        rs.changed({"sample_cap": 5})
    with pytest.raises(ValueError, match="true or false"):
        rs.changed({"ask_require_approval": "no"})
