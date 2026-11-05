"""OTLP/HTTP logs intake (OpenTelemetry): ``POST /v1/logs`` with an ExportLogsServiceRequest,
JSON or protobuf, from logpipe-agent, the OpenTelemetry Collector or any OTel SDK.

Each log record becomes one intake event. Semantic-convention attributes give its host
(``host.name``), service (``service.name``) and source (``logpipe.source``, else the file name,
else the instrumentation scope); ``severity`` becomes a ``severity:<level>`` tag, which the
level facet trusts over guessing from the text.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest

PROTOBUF = "application/x-protobuf"


def decode(raw: bytes, content_type: str) -> dict:
    """The request as OTLP JSON (protobuf is converted to the same shape)."""
    if content_type.split(";")[0].strip() == PROTOBUF:
        request = ExportLogsServiceRequest()
        try:
            request.ParseFromString(raw)
        except Exception as error:  # DecodeError and friends
            raise ValueError(f"not an OTLP ExportLogsServiceRequest: {error}") from error
        return MessageToDict(request)
    try:
        body = json.loads(raw)
    except ValueError as error:
        raise ValueError(f"not JSON: {error}") from error
    if not isinstance(body, dict) or not isinstance(body.get("resourceLogs", []), list):
        raise ValueError('expected an OTLP logs request: {"resourceLogs": [...]}')  # noqa: TRY004 - shown as a 400
    return body


def _value(v: dict | None):
    """An OTLP AnyValue as plain Python."""
    if not v:
        return None
    for key in ("stringValue", "boolValue", "doubleValue"):
        if key in v:
            return v[key]
    if "intValue" in v:
        return int(v["intValue"])
    if "arrayValue" in v:
        return [_value(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return _attrs(v["kvlistValue"].get("values", []))
    if "bytesValue" in v:
        return v["bytesValue"]
    return None


def _attrs(items: list | None) -> dict:
    return {a.get("key", ""): _value(a.get("value")) for a in items or []}


def _level(record: dict) -> str | None:
    """OTel severity -> logpipe's error/warn/info/debug (SeverityNumber ranges, else the text)."""
    number = record.get("severityNumber")
    if isinstance(number, str):  # protobuf JSON gives the enum name, e.g. SEVERITY_NUMBER_WARN
        name = number.removeprefix("SEVERITY_NUMBER_").rstrip("0123456789").lower()
        return {"trace": "debug", "debug": "debug", "info": "info", "warn": "warn",
                "error": "error", "fatal": "error"}.get(name)
    if isinstance(number, int) and number > 0:
        return "debug" if number <= 8 else "info" if number <= 12 else "warn" if number <= 16 else "error"
    text = str(record.get("severityText") or "").lower()
    for prefix, lvl in (("fatal", "error"), ("crit", "error"), ("err", "error"), ("warn", "warn"),
                        ("info", "info"), ("notice", "info"), ("debug", "debug"), ("trace", "debug")):
        if text.startswith(prefix):
            return lvl
    return None


def _zone(offset: str | None):
    """``+08:00`` -> that zone (the agent host's), else UTC."""
    try:
        sign = -1 if offset and offset[0] == "-" else 1
        hours, minutes = (offset or "")[1:].split(":")
        return timezone(sign * timedelta(hours=int(hours), minutes=int(minutes)))
    except (ValueError, IndexError):
        return UTC


def events(request: dict) -> list[dict]:
    """Intake events (``{ts, host, service, source, tags, message}``) for every log record."""
    out = []
    for rl in request.get("resourceLogs") or []:
        res = _attrs((rl.get("resource") or {}).get("attributes"))
        zone = _zone(res.get("logpipe.host.utc_offset"))
        for sl in rl.get("scopeLogs") or []:
            scope = (sl.get("scope") or {}).get("name") or ""
            for rec in sl.get("logRecords") or []:
                body = _value(rec.get("body"))
                if body is None or body == "":
                    continue
                attrs = _attrs(rec.get("attributes"))
                nanos = int(rec.get("timeUnixNano") or 0) or int(rec.get("observedTimeUnixNano") or 0)
                ts = (datetime.fromtimestamp(nanos / 1e9, tz=UTC).astimezone(zone) if nanos
                      else datetime.now(UTC)).isoformat()
                tags = [str(t) for t in (attrs.get("logpipe.tags") or res.get("logpipe.tags") or [])]
                if level := _level(rec):
                    tags.append(f"severity:{level}")
                out.append({
                    "ts": ts,
                    "host": attrs.get("host.name") or res.get("host.name") or "unknown",
                    "service": res.get("service.name") or attrs.get("service.name") or "",
                    "source": (res.get("logpipe.source") or attrs.get("log.file.name")
                               or scope or "otlp"),
                    "tags": tags,
                    "message": body if isinstance(body, str) else json.dumps(body, default=str),
                })
    return out
