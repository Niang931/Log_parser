"""Synthetic IoT fleet with labelled anomalies and a firmware rollout.

No public IoT log corpus carries anomaly labels, so this generator plays a
fleet of thermostats, door locks, smart plugs and gateways.  Each device
runs a small state machine that produces realistic causal orderings
(keypad accepted → unlocked → door open → door closed → locked).

``generate`` returns events for three phases:

* ``history``: normal behaviour on firmware 1.x (calibration + training).
* ``validation``: normal behaviour with injected anomalies (threshold tuning).
* ``test``: same as validation, plus a mid-way firmware rollout that
  changes wording, adds fields and introduces new normal event types.

Injected anomaly kinds, labelled per event:

* ``reorder``: a causal pair swapped (door opens before the unlock).
* ``missing``: a step skipped (unlocked without an accepted code).
* ``new_event``: a fault message never seen before.
* ``burst``: one event repeated many times (keypad brute force).
* ``stuck``: a relay flapping on/off with no telemetry between.
"""
from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..sources.file_replay import LogEvent

ANOMALY_KINDS = ("reorder", "missing", "new_event", "burst", "stuck")
_FAULTS = [
    "Watchdog reset reason=0x{code:02X} uptime={n}s",
    "Sensor fault code=E{code} retrying in {n}ms",
    "Flash write failed sector={code} errno=-{n}",
    "Brownout detected vbat={n}mV threshold=3300mV",
    "TLS handshake failed alert={code} peer=broker.local",
]


@dataclass(frozen=True)
class SynthEvent:
    device_id: str
    timestamp: datetime
    line: str
    phase: str
    anomaly: str | None = None
    firmware: int = 1

    def to_log_event(self, line_no: int = 0) -> LogEvent:
        return LogEvent(self.device_id, f"synth:{self.phase}", line_no, self.line)


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


class _Device:
    kind = "device"

    def __init__(self, device_id: str, rng: random.Random):
        self.id = device_id
        self.rng = rng
        self.firmware = 1

    def line(self, now: datetime, level: str, message: str) -> str:
        return f"{_ts(now)} [{level}] {message}"

    def episode(self) -> list[tuple[str, str]]:
        """One causal episode as (level, message) pairs."""
        raise NotImplementedError

    def fault(self) -> str:
        template = self.rng.choice(_FAULTS)
        return template.format(code=self.rng.randint(1, 250), n=self.rng.randint(10, 9000))


class Thermostat(_Device):
    kind = "thermostat"

    def __init__(self, device_id, rng):
        super().__init__(device_id, rng)
        self.temp, self.target, self.heating = 21.0, 22.5, False

    def episode(self):
        r = self.rng
        self.temp += (0.25 if self.heating else -0.15) + r.uniform(-0.05, 0.05)
        mode = r.choice(["auto", "auto", "auto", "heat", "eco"])
        hum = r.randint(35, 60)
        tele = (f"Thermostat SN:{self.id} temp={self.temp:.1f}C target={self.target:.1f}C "
                f"mode={mode} humidity={hum}%")
        if self.firmware >= 2:
            tele += " fw=2.1.0"
        out = [("INFO", tele)]
        if not self.heating and self.temp < self.target - 0.3:
            self.heating = True
            out.append(("INFO", f"Heater relay switched state=on duty={r.randint(20, 90)}"))
        elif self.heating and self.temp >= self.target:
            self.heating = False
            out.append(("INFO", "Heater relay switched state=off duty=0"))
        out.append(("INFO", f"MQTT publish topic=home/{self.id}/telemetry qos=1 bytes={r.randint(180, 260)}"))
        if r.random() < 0.05:
            out.append(("WARN", f"Wifi rssi={r.randint(-85, -71)} dBm below threshold -70 dBm"))
        if self.firmware >= 2 and r.random() < 0.2:
            out.append(("INFO", f"Schedule applied profile={r.choice(['home', 'away', 'night'])} slot={r.randint(0, 7)}"))
        return out


class DoorLock(_Device):
    kind = "doorlock"

    def __init__(self, device_id, rng):
        super().__init__(device_id, rng)
        self.battery = rng.randint(60, 100)

    def episode(self):
        r = self.rng
        if r.random() < 0.6:
            self.battery = max(5, self.battery - (1 if r.random() < 0.05 else 0))
            return [("INFO", f"Lock {self.id} heartbeat battery={self.battery}% rssi={r.randint(-80, -40)}")]
        user, slot = r.choice(["alice", "bob", "carol", "dave"]), r.randint(1, 8)
        out = [
            ("INFO", f"Keypad code accepted user={user} slot={slot}"),
            ("INFO", f"Lock {self.id} state=unlocked battery={self.battery}% user={user}"),
            ("INFO", "Door sensor contact=open"),
            ("INFO", "Door sensor contact=closed"),
            ("INFO", f"Lock {self.id} state=locked battery={self.battery}% user=auto"),
            ("INFO", f"MQTT publish topic=home/{self.id}/events qos=1 bytes={r.randint(80, 120)}"),
        ]
        if self.firmware >= 2:
            out.insert(4, ("INFO", f"Auto-relock timer expired after {r.randint(5, 30)}s"))
        return out


class SmartPlug(_Device):
    kind = "plug"

    def __init__(self, device_id, rng):
        super().__init__(device_id, rng)
        self.on = False

    def episode(self):
        r = self.rng
        out = []
        if r.random() < 0.15:
            self.on = not self.on
            out.append(("INFO", f"Relay toggled state={'on' if self.on else 'off'} source={r.choice(['app', 'schedule', 'button'])}"))
        watts = r.uniform(20, 1800) if self.on else r.uniform(0, 1.5)
        out.append(("INFO", f"Plug {self.id} power={watts:.1f}W voltage={r.uniform(228, 242):.1f}V current={watts / 235:.2f}A"))
        return out


class Gateway(_Device):
    kind = "gateway"

    def episode(self):
        r = self.rng
        out = [("INFO", f"Gateway {self.id} connected clients={r.randint(3, 40)} uptime={r.randint(1000, 900000)}s")]
        if r.random() < 0.1:
            version = "2.1.0" if self.firmware >= 2 else "1.4.2"
            out.append(("INFO", f"OTA check version={version} available=false"))
        if r.random() < 0.05:
            out.append(("INFO", f"MQTT broker keepalive ok latency={r.randint(2, 80)}ms"))
        return out


KINDS = [Thermostat, DoorLock, SmartPlug, Gateway]
_PREFIX = {"thermostat": "TH", "doorlock": "DL", "plug": "PL", "gateway": "GW"}


def _inject(device: _Device, kind: str, episode: list[tuple[str, str]]) -> list[tuple[str, str, bool]]:
    """Turn a normal episode into an anomalous one; the bool marks anomalous events."""
    r = device.rng
    normal = [(lvl, msg, False) for lvl, msg in episode]
    if kind == "new_event":
        pos = r.randint(0, len(normal))
        return normal[:pos] + [("ERROR", device.fault(), True)] + normal[pos:]
    if kind == "burst":
        lvl, msg = episode[0]
        if isinstance(device, DoorLock):
            lvl, msg = "WARN", "Keypad code rejected user=unknown slot=0"
        return [(lvl, msg, True)] * r.randint(8, 15) + normal
    if kind == "stuck":
        flaps = []
        for i in range(r.randint(8, 14)):
            state = "on" if i % 2 == 0 else "off"
            msg = (f"Relay toggled state={state} source=button" if isinstance(device, SmartPlug)
                   else f"Heater relay switched state={state} duty={40 if state == 'on' else 0}")
            flaps.append(("INFO", msg, True))
        return flaps + normal
    if kind == "reorder" and len(normal) >= 3:
        i = r.randrange(len(normal) - 1)
        swapped = list(normal)
        a, b = swapped[i], swapped[i + 1]
        swapped[i], swapped[i + 1] = (b[0], b[1], True), (a[0], a[1], True)
        return swapped
    if kind == "missing" and len(normal) >= 3:
        i = r.randrange(1, len(normal) - 1)
        out = list(normal)
        del out[i]
        nxt = out[i]
        out[i] = (nxt[0], nxt[1], True)
        return out
    # Episode too short for this kind: fall back to a fault message.
    return _inject(device, "new_event", episode)


def _anomaly_kind(device: _Device, rng: random.Random) -> str:
    options = ["new_event", "burst"]
    if isinstance(device, DoorLock):
        options += ["reorder", "missing", "reorder", "missing"]
    if isinstance(device, (SmartPlug, Thermostat)):
        options += ["stuck"]
    return rng.choice(options)


def _run_phase(
    devices: list[_Device], phase: str, start: datetime, hours: float, rng: random.Random,
    anomaly_rate: float, rollout_at: float | None, rollout_share: float,
) -> list[SynthEvent]:
    events: list[SynthEvent] = []
    end = start + timedelta(hours=hours)
    rollout_time = start + timedelta(hours=hours * rollout_at) if rollout_at is not None else None
    upgrade = {d.id for d in devices if rng.random() < rollout_share}
    for device in devices:
        now = start + timedelta(seconds=rng.uniform(0, 30))
        while now < end:
            if rollout_time is not None and now >= rollout_time and device.id in upgrade:
                device.firmware = 2
            episode = device.episode()
            if anomaly_rate and rng.random() < anomaly_rate:
                kind = _anomaly_kind(device, rng)
                tagged = [(lvl, msg, kind if bad else None) for lvl, msg, bad in _inject(device, kind, episode)]
            else:
                tagged = [(lvl, msg, None) for lvl, msg in episode]
            for lvl, msg, tag in tagged:
                now += timedelta(milliseconds=rng.randint(40, 900))
                now = now.replace(microsecond=now.microsecond // 1000 * 1000)
                events.append(SynthEvent(device.id, now, device.line(now, lvl, msg), phase, tag, device.firmware))
            now += timedelta(seconds=rng.uniform(5, 25))
    events.sort(key=lambda e: (e.timestamp, e.device_id))
    return events


@dataclass
class Fleet:
    history: list[SynthEvent]
    validation: list[SynthEvent]
    test: list[SynthEvent]

    def write(self, out: Path) -> None:
        """One file per device and phase, plus a TSV of anomalous lines."""
        for phase in ("history", "validation", "test"):
            by_device: dict[str, list[str]] = {}
            for e in getattr(self, phase):
                by_device.setdefault(e.device_id, []).append(e.line)
            (out / phase).mkdir(parents=True, exist_ok=True)
            for device, lines in by_device.items():
                (out / phase / f"{device}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with (out / "labels.tsv").open("w", encoding="utf-8") as fh:
            fh.write("phase\tdevice_id\ttimestamp\tanomaly\tline\n")
            for phase in ("validation", "test"):
                for e in getattr(self, phase):
                    if e.anomaly:
                        fh.write(f"{phase}\t{e.device_id}\t{e.timestamp.isoformat()}\t{e.anomaly}\t{e.line}\n")


class Stream:
    """Live load: ``rate`` lines/s from ``devices`` devices, stamped with the wall clock.

    Iterate to get ``LogEvent``s. ``generated`` counts the lines yielded so far, so a run can
    be checked by counting. ``rollout_at_s`` seconds in, ``rollout_share`` of the devices
    switch to firmware 2 (as in ``_run_phase``). Stops after ``duration_s`` (never if None).
    """

    def __init__(
        self, devices: int, rate: float, rollout_at_s: float | None = None,
        rollout_share: float = 0.6, seed: int = 7, duration_s: float | None = None,
    ):
        if devices < 1 or rate <= 0:
            raise ValueError(f"need devices >= 1 and rate > 0, got {devices=} {rate=}")
        rng = random.Random(seed)
        self.devices: list[_Device] = []
        for i in range(devices):
            kind = KINDS[i % len(KINDS)]
            device_id = f"{_PREFIX[kind.kind]}-{1000 + i // len(KINDS)}"
            self.devices.append(kind(device_id, random.Random(rng.random())))
        self._upgrade = [d for d in self.devices if rng.random() < rollout_share]
        self._rng = rng
        self.rate, self.rollout_at_s, self.duration_s = rate, rollout_at_s, duration_s
        self.generated = 0

    def __iter__(self) -> Iterator[LogEvent]:
        # The lines of each device's current episode, so one device keeps its causal order.
        pending: dict[str, deque[tuple[str, str]]] = {d.id: deque() for d in self.devices}
        start = time.monotonic()
        rolled_out = False
        sent = 0  # per iteration, so iterating again starts a fresh schedule
        while True:
            offset = sent / self.rate  # when this line is due
            if self.duration_s is not None and offset >= self.duration_s:
                return
            # ponytail: sleeps only when >1 ms ahead; behind schedule it emits back to back
            if (ahead := start + offset - time.monotonic()) > 0.001:
                time.sleep(ahead)
            if (not rolled_out and self.rollout_at_s is not None
                    and time.monotonic() - start >= self.rollout_at_s):
                for device in self._upgrade:
                    device.firmware = 2
                rolled_out = True
            device = self._rng.choice(self.devices)
            if not pending[device.id]:
                pending[device.id].extend(device.episode())
            level, message = pending[device.id].popleft()
            line = device.line(datetime.now(UTC), level, message)
            sent += 1
            self.generated += 1
            yield LogEvent(device.id, "synth:load", self.generated, line)


def generate(
    devices_per_kind: int = 5,
    history_hours: float = 12,
    validation_hours: float = 3,
    test_hours: float = 6,
    anomaly_rate: float = 0.01,
    rollout_at: float = 0.5,
    rollout_share: float = 0.6,
    seed: int = 7,
) -> Fleet:
    rng = random.Random(seed)
    devices: list[_Device] = []
    for kind in KINDS:
        for i in range(devices_per_kind):
            devices.append(kind(f"{_PREFIX[kind.kind]}-{1000 + i}", random.Random(rng.random())))
    t0 = datetime(2026, 4, 13, tzinfo=UTC)
    history = _run_phase(devices, "history", t0, history_hours, rng, 0.0, None, 0.0)
    t1 = t0 + timedelta(hours=history_hours)
    validation = _run_phase(devices, "validation", t1, validation_hours, rng, anomaly_rate, None, 0.0)
    t2 = t1 + timedelta(hours=validation_hours)
    test = _run_phase(devices, "test", t2, test_hours, rng, anomaly_rate, rollout_at, rollout_share)
    return Fleet(history, validation, test)


def as_log_events(events: list[SynthEvent]) -> Iterator[LogEvent]:
    for i, e in enumerate(events):
        yield e.to_log_event(i)
