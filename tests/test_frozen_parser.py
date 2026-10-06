import threading

from logpipe.calibrate import calibrate
from logpipe.enrich.llm import OfflineAdapter
from logpipe.parsing.frozen_parser import FrozenParser, ParserHandle
from logpipe.registry import UNK_TEMPLATE_ID, NewTemplate, SchemaRegistry

NEW_LINE = "2026-04-13 10:20:00.000 [ERROR] OTA update failed code=0x1F retry=3"


def test_calibration_lines_all_parse_final(snapshot, fixture_lines):
    parser = FrozenParser(snapshot)
    results = [parser.parse(line) for line in fixture_lines]
    assert all(r.status == "FINAL" for r in results)
    assert all(r.registry_version == snapshot.version for r in results)


def test_thermostat_readings_share_one_template_with_named_keys(snapshot, fixture_lines):
    parser = FrozenParser(snapshot)
    thermo = [parser.parse(line) for line in fixture_lines if "Thermostat" in line]
    assert len({r.template_id for r in thermo}) == 1
    assert thermo[0].payload["temp"] == "22.5C"
    assert thermo[0].payload["humidity"] == "45%"


def test_opposite_states_stay_distinct_event_types(snapshot):
    parser = FrozenParser(snapshot)
    opened = parser.parse("2026-04-13 11:00:00.000 [INFO] Door sensor contact=open")
    closed = parser.parse("2026-04-13 11:00:01.000 [INFO] Door sensor contact=closed")
    assert opened.status == closed.status == "FINAL"
    assert opened.template_id != closed.template_id


def test_unknown_line_is_pending_and_resolves_after_swap(snapshot):
    handle = ParserHandle(snapshot)
    pending = handle.parse(NEW_LINE)
    assert (pending.template_id, pending.status, pending.payload) == (UNK_TEMPLATE_ID, "PENDING", {})

    registry = SchemaRegistry(snapshot)
    learned = calibrate([NEW_LINE], OfflineAdapter())
    (entry,) = learned.templates.values()
    new_snapshot, (new_id,) = registry.publish([NewTemplate(entry.tokens, entry.keys)])
    handle.swap(new_snapshot)

    resolved = handle.parse(NEW_LINE)
    assert (resolved.template_id, resolved.status) == (new_id, "FINAL")
    assert resolved.payload["retry"] == "3"


def test_swap_ignores_older_snapshots(snapshot):
    newer, _ = SchemaRegistry(snapshot).publish([])
    handle = ParserHandle(newer)
    handle.swap(snapshot)
    assert handle.version == newer.version


def test_concurrent_parse_during_swaps_sees_only_whole_versions(snapshot, fixture_lines):
    registry = SchemaRegistry(snapshot)
    handle = ParserHandle(snapshot)
    errors: list[BaseException] = []
    seen: set[int] = set()
    stop = threading.Event()

    def reader():
        try:
            while not stop.is_set():
                for line in fixture_lines:
                    result = handle.parse(line)
                    assert result.status == "FINAL"
                    seen.add(result.registry_version)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(20):
        new_snapshot, _ = registry.publish([NewTemplate((f"synthetic-{i}",), ())])
        handle.swap(new_snapshot)
    stop.set()
    for t in threads:
        t.join()

    assert not errors
    assert seen <= set(range(snapshot.version, registry.current.version + 1))
