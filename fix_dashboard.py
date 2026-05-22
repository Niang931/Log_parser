import json
import pathlib

dashboard = {
  "title": "DeepParse v2",
  "uid": "deepparse-main",
  "panels": [
    {
      "id": 1,
      "title": "All Pipeline Events",
      "type": "logs",
      "gridPos": {"x":0,"y":0,"w":24,"h":8},
      "options": {"showTime": True, "sortOrder": "Descending", "wrapLogMessage": True},
      "targets": [{"datasource": {"type": "loki", "uid": "loki"}, "expr": "{app=\"deepparse\"}"}]
    },
    {
      "id": 2,
      "title": "Run Complete Events",
      "type": "logs",
      "gridPos": {"x":0,"y":8,"w":24,"h":8},
      "options": {"showTime": True, "sortOrder": "Descending", "wrapLogMessage": True},
      "targets": [{"datasource": {"type": "loki", "uid": "loki"}, "expr": "{app=\"deepparse\", event_type=\"run_complete\"}"}]
    },
    {
      "id": 3,
      "title": "Adaptive Loop Probes",
      "type": "logs",
      "gridPos": {"x":0,"y":16,"w":12,"h":6},
      "options": {"showTime": True, "sortOrder": "Descending"},
      "targets": [{"datasource": {"type": "loki", "uid": "loki"}, "expr": "{app=\"deepparse\", event_type=\"probe\"}"}]
    },
    {
      "id": 4,
      "title": "LLM Call Events",
      "type": "logs",
      "gridPos": {"x":12,"y":16,"w":12,"h":6},
      "options": {"showTime": True, "sortOrder": "Descending"},
      "targets": [{"datasource": {"type": "loki", "uid": "loki"}, "expr": "{app=\"deepparse\", event_type=\"llm_call\"}"}]
    },
    {
      "id": 5,
      "title": "Security - Injection Attempts",
      "type": "logs",
      "gridPos": {"x":0,"y":22,"w":12,"h":6},
      "options": {"showTime": True},
      "targets": [{"datasource": {"type": "loki", "uid": "loki"}, "expr": "{app=\"deepparse\", event_type=\"security\"}"}]
    },
    {
      "id": 6,
      "title": "Warnings and Errors",
      "type": "logs",
      "gridPos": {"x":12,"y":22,"w":12,"h":6},
      "options": {"showTime": True},
      "targets": [{"datasource": {"type": "loki", "uid": "loki"}, "expr": "{app=\"deepparse\", level=\"warn\"}"}]
    }
  ],
  "time": {"from": "now-6h", "to": "now"},
  "refresh": "10s",
  "schemaVersion": 38
}

# Write deepparse.json with no BOM
path = pathlib.Path("grafana/provisioning/dashboards/deepparse.json")
path.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
print("deepparse.json written OK")

# Write loki.yaml with no BOM
loki_yaml = """apiVersion: 1
datasources:
  - name: Loki
    type: loki
    uid: loki
    access: proxy
    url: http://loki:3100
    isDefault: true
    jsonData:
      maxLines: 1000
"""
pathlib.Path("grafana/provisioning/datasources/loki.yaml").write_text(
    loki_yaml, encoding="utf-8"
)
print("loki.yaml written OK")

# Verify no BOM in either file
for f in ["grafana/provisioning/dashboards/deepparse.json",
          "grafana/provisioning/datasources/loki.yaml"]:
    raw = open(f, "rb").read(4)
    bom = raw.startswith(b"\xef\xbb\xbf")
    print(f"{f}: BOM={'YES - BAD' if bom else 'NO - GOOD'}")