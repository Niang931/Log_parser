"""
fix_dashboard.py — SELIS dashboard, clean rebuild
Fixes:
  - Adaptive Loop dual tile: use sum by() to force single stream
  - Schema Drift no data: or vector(0) fallback not in LogQL, use noValue + empty stream push
  - All counts use $__range
  - noValue:"0" on every stat panel
"""
import json, pathlib

DS = {"type": "loki", "uid": "loki"}

pathlib.Path("grafana/provisioning/datasources/loki.yaml").write_text(
    "apiVersion: 1\ndatasources:\n"
    "  - name: Loki\n    type: loki\n    uid: loki\n"
    "    access: proxy\n    url: http://loki:3100\n"
    "    isDefault: true\n    jsonData:\n      maxLines: 5000\n",
    encoding="utf-8")
print("loki.yaml OK")


def stat(pid, title, expr, steps_or_color, x, y, w=4, h=4, desc="", link=None):
    if isinstance(steps_or_color, str):
        color_cfg = {"mode": "fixed", "fixedColor": steps_or_color}
        steps     = [{"color": steps_or_color, "value": 0}]
    else:
        color_cfg = {"mode": "thresholds"}
        steps     = steps_or_color

    fd = {
        "noValue": "0",
        "color":   color_cfg,
        "thresholds": {"mode": "absolute", "steps": steps},
        "mappings": [],
    }
    if link:
        fd["links"] = [{"title": "View logs", "url": link}]

    return {
        "id": pid, "title": title, "description": desc,
        "type": "stat",
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "options": {
            "colorMode": "background", "graphMode": "none",
            "textMode": "auto",
            "reduceOptions": {
                "calcs": ["lastNotNull"],
                "fields": "/^Value$/",   # ← only pick the Value field, not legend labels
                "values": False,
            },
        },
        "fieldConfig": {"defaults": fd},
        "targets": [{
            "datasource": DS,
            # sum by() with empty by-clause forces exactly one output stream
            "expr": expr,
            "instant": True,
            "range": False,
            "legendFormat": "{{event_type}}",
            "refId": "A",
        }],
    }


def ts(pid, title, targets, x, y, w=12, h=8):
    return {
        "id": pid, "title": title, "type": "timeseries",
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "options": {
            "tooltip": {"mode": "multi"},
            "legend":  {"displayMode": "list", "placement": "bottom"},
        },
        "fieldConfig": {"defaults": {
            "custom": {"lineWidth": 2, "fillOpacity": 12},
            "color": {"mode": "palette-classic"},
        }},
        "targets": [
            {**t, "datasource": DS,
             "instant": False, "range": True, "refId": chr(65+i)}
            for i, t in enumerate(targets)
        ],
    }


def logs(pid, title, expr, x, y, w=12, h=7, desc=""):
    return {
        "id": pid, "title": title, "description": desc,
        "type": "logs",
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "options": {
            "showTime": True, "sortOrder": "Descending",
            "wrapLogMessage": True, "enableLogDetails": True,
        },
        "targets": [{"datasource": DS, "expr": expr,
                     "instant": False, "range": True, "refId": "A"}],
    }


def row(pid, title, y):
    return {"id": pid, "title": title, "type": "row",
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
            "collapsed": False}


# ── Count helper — sum by() forces single output value ───────────────────
def cnt(event_type):
    return (f'sum by() '
            f'(count_over_time({{app="deepparse",'
            f'event_type="{event_type}"}}[$__range]))')


panels = []

# ── Row 1 — KPI counters (y=0) ────────────────────────────────────────────
panels.append(row(1, "KPI Counters", 0))

panels.append(stat(2, "Total Runs", cnt("run_complete"),
    "blue", 0, 1,
    desc="Completed pipeline runs in the selected time window"))

panels.append(stat(3, "LLM Fallback Calls", cnt("llm_call"),
    [{"color":"green","value":0},
     {"color":"yellow","value":5},
     {"color":"orange","value":20}],
    4, 1,
    desc="Offline path: novel tokens sent to Gemini LLM"))

panels.append(stat(4, "Injections Blocked", cnt("security"),
    [{"color":"green","value":0},{"color":"red","value":1}],
    8, 1,
    desc="Prompt injection attempts blocked by sanitisation pipeline"))

panels.append(stat(5, "Human Queue", cnt("human_queue"),
    [{"color":"green","value":0},
     {"color":"orange","value":1},
     {"color":"red","value":10}],
    12, 1,
    desc="Lines neither Drain nor LLM resolved — click to see panel 18",
    link="/d/deepparse-main?orgId=1&viewPanel=18"))

panels.append(stat(6, "Adaptive Loop Rounds", cnt("probe"),
    [{"color":"green","value":0},
     {"color":"yellow","value":10},
     {"color":"red","value":30}],
    16, 1,
    desc="Loop 1: parse-rate probes fired during mask synthesis"))

panels.append(stat(7, "Schema Drift Alerts", cnt("schema_drift"),
    [{"color":"green","value":0},{"color":"red","value":1}],
    20, 1,
    desc="0 means no drift detected (healthy). Wipe Loki and re-run to reset."))

# ── Row 2 — Trend charts (y=6) ────────────────────────────────────────────
panels.append(row(8, "Trend Charts", 6))

panels.append(ts(9, "Pipeline Runs Over Time",
    [{"expr": 'count_over_time({app="deepparse",event_type="run_complete"}[$__interval])',
      "legendFormat": "runs"}],
    0, 7, 12, 8))

panels.append(ts(10, "LLM Calls vs Adaptive Probes",
    [{"expr": 'count_over_time({app="deepparse",event_type="llm_call"}[$__interval])',
      "legendFormat": "LLM calls"},
     {"expr": 'count_over_time({app="deepparse",event_type="probe"}[$__interval])',
      "legendFormat": "Loop 1 probes"}],
    12, 7, 12, 8))

# ── Row 3 — Run events (y=16) ─────────────────────────────────────────────
panels.append(row(11, "Run Events", 16))

panels.append(logs(12,
    "Run Complete — Parse Rate · Masks · Templates · Elapsed",
    '{app="deepparse",event_type="run_complete"}',
    0, 17, 24, 7,
    desc="One entry per run: parse_rate, log_count, llm_calls, unique_templates"))

# ── Row 4 — Offline/Online path (y=25) ───────────────────────────────────
panels.append(row(13, "Offline / Online Path Events", 25))

panels.append(logs(14,
    "LLM Fallback Events (Offline — Gemini resolves novel tokens)",
    '{app="deepparse",event_type="llm_call"}',
    0, 26, 12, 7))

panels.append(logs(15,
    "Adaptive Loop Probes (Loop 1 — parse-rate feedback)",
    '{app="deepparse",event_type="probe"}',
    12, 26, 12, 7))

# ── Row 5 — Security + Human queue (y=34) ────────────────────────────────
panels.append(row(16, "Security & Human Queue", 34))

panels.append(logs(17,
    "Security — Injection Sanitisation Pipeline",
    '{app="deepparse",event_type="security"}',
    0, 35, 12, 7))

panels.append(logs(18,
    "Human Queue — Operator Review Required (click Human Queue tile)",
    '{app="deepparse",event_type="human_queue"}',
    12, 35, 12, 7,
    desc="raw: original line · reason: why it failed · operator must add to template library"))

# ── Row 6 — Warnings + raw (y=43) ────────────────────────────────────────
panels.append(row(19, "Warnings & Raw Stream", 43))

panels.append(logs(20,
    "Warnings — Low Parse Rate · Rate Limits · LLM Errors",
    '{app="deepparse",level="warn"}',
    0, 44, 12, 7))

panels.append(logs(21,
    "All Pipeline Events (raw stream)",
    '{app="deepparse"}',
    12, 44, 12, 7))

dashboard = {
    "title":         "SELIS \u2014 Self-Evolving Log Intelligence System",
    "uid":           "deepparse-main",
    "tags":          ["selis","deepparse","fab"],
    "time":          {"from": "now-1h", "to": "now"},
    "timepicker":    {},
    "refresh":       "10s",
    "schemaVersion": 38,
    "version":       1,
    "panels":        panels,
}

path = pathlib.Path("grafana/provisioning/dashboards/deepparse.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
json.loads(path.read_text(encoding="utf-8"))
raw = open(path,"rb").read(4)
print(f"deepparse.json  BOM={'BAD' if raw.startswith(b'\\xef\\xbb\\xbf') else 'OK'}  JSON=valid")
print(f"  title   : {dashboard['title']}")
print(f"  panels  : {len(panels)}")
print(f"  noValue : 0 on all stat panels")
print(f"  sum by(): single value per stat panel (no more dual tiles)")
print()
print("Steps to get clean counts:")
print("  1.  docker compose down -v      (wipes Loki data)")
print("  2.  docker compose up -d")
print("  3.  python fix_dashboard.py")
print("  4.  docker compose restart grafana")
print("  5.  python main.py / run demo   (fresh events only)")