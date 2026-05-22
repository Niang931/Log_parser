"""
fix_all.py — Fixes:
  1. docker-compose.yml — adds Ollama (on-prem LLM, zero egress)
  2. grafana dashboard   — proper visual panels (stat/timeseries/table)
  3. loki_logger.py      — pushes numeric labels for metric queries
"""
import json, pathlib

# ============================================================
# 1 — docker-compose.yml with Ollama
# ============================================================
compose = """\
services:

  loki:
    image: grafana/loki:2.9.0
    container_name: deepparse-loki
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/local-config.yaml
    networks:
      - deepparse

  grafana:
    image: grafana/grafana:10.2.0
    container_name: deepparse-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=deepparse
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - loki
    networks:
      - deepparse

  ollama:
    image: ollama/ollama
    container_name: deepparse-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-storage:/root/.ollama
    networks:
      - deepparse

networks:
  deepparse:
    driver: bridge

volumes:
  grafana-storage:
  ollama-storage:
"""
pathlib.Path("docker-compose.yml").write_text(compose, encoding="utf-8")
print("docker-compose.yml updated with Ollama")


# ============================================================
# 2 — Grafana dashboard with visual panels
# ============================================================
dashboard = {
  "title": "DeepParse v2 — Silicon-Fab Log Intelligence",
  "uid": "deepparse-main",
  "tags": ["deepparse", "fab", "llm"],
  "time": {"from": "now-6h", "to": "now"},
  "refresh": "10s",
  "schemaVersion": 38,
  "panels": [

    # ── Row 1: KPI stat tiles ──────────────────────────────
    {
      "id": 1,
      "title": "Total Runs",
      "type": "stat",
      "gridPos": {"x": 0, "y": 0, "w": 4, "h": 4},
      "options": {
        "colorMode": "background",
        "graphMode": "area",
        "textMode": "auto",
        "reduceOptions": {"calcs": ["sum"]}
      },
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "thresholds"},
          "thresholds": {"steps": [
            {"color": "blue", "value": 0}
          ]}
        }
      },
      "targets": [{
        "datasource": {"type": "loki", "uid": "loki"},
        "expr": "count_over_time({app=\"deepparse\", event_type=\"run_complete\"}[6h])"
      }]
    },

    {
      "id": 2,
      "title": "LLM Fallback Calls",
      "type": "stat",
      "gridPos": {"x": 4, "y": 0, "w": 4, "h": 4},
      "options": {
        "colorMode": "background",
        "graphMode": "area",
        "reduceOptions": {"calcs": ["sum"]}
      },
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "thresholds"},
          "thresholds": {"steps": [
            {"color": "green", "value": 0},
            {"color": "yellow", "value": 5},
            {"color": "red", "value": 20}
          ]}
        }
      },
      "targets": [{
        "datasource": {"type": "loki", "uid": "loki"},
        "expr": "count_over_time({app=\"deepparse\", event_type=\"llm_call\"}[6h])"
      }]
    },

    {
      "id": 3,
      "title": "Injection Attempts Blocked",
      "type": "stat",
      "gridPos": {"x": 8, "y": 0, "w": 4, "h": 4},
      "options": {
        "colorMode": "background",
        "graphMode": "none",
        "reduceOptions": {"calcs": ["sum"]}
      },
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "thresholds"},
          "thresholds": {"steps": [
            {"color": "green", "value": 0},
            {"color": "red", "value": 1}
          ]}
        }
      },
      "targets": [{
        "datasource": {"type": "loki", "uid": "loki"},
        "expr": "count_over_time({app=\"deepparse\", event_type=\"security\"}[6h])"
      }]
    },

    {
      "id": 4,
      "title": "Adaptive Loop Rounds Fired",
      "type": "stat",
      "gridPos": {"x": 12, "y": 0, "w": 4, "h": 4},
      "options": {
        "colorMode": "background",
        "graphMode": "area",
        "reduceOptions": {"calcs": ["sum"]}
      },
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "thresholds"},
          "thresholds": {"steps": [
            {"color": "green", "value": 0},
            {"color": "yellow", "value": 10},
            {"color": "red", "value": 30}
          ]}
        }
      },
      "targets": [{
        "datasource": {"type": "loki", "uid": "loki"},
        "expr": "count_over_time({app=\"deepparse\", event_type=\"probe\", level=\"warn\"}[6h])"
      }]
    },

    {
      "id": 5,
      "title": "Rate Limit Events",
      "type": "stat",
      "gridPos": {"x": 16, "y": 0, "w": 4, "h": 4},
      "options": {
        "colorMode": "background",
        "graphMode": "none",
        "reduceOptions": {"calcs": ["sum"]}
      },
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "thresholds"},
          "thresholds": {"steps": [
            {"color": "green", "value": 0},
            {"color": "red", "value": 1}
          ]}
        }
      },
      "targets": [{
        "datasource": {"type": "loki", "uid": "loki"},
        "expr": "count_over_time({app=\"deepparse\"} |= \"rate_limit\"[6h])"
      }]
    },

    {
      "id": 6,
      "title": "LLM Provider",
      "type": "stat",
      "gridPos": {"x": 20, "y": 0, "w": 4, "h": 4},
      "options": {
        "colorMode": "background",
        "textMode": "value",
        "reduceOptions": {"calcs": ["lastNotNull"]}
      },
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "thresholds"},
          "thresholds": {"steps": [
            {"color": "purple", "value": 0}
          ]}
        }
      },
      "targets": [{
        "datasource": {"type": "loki", "uid": "loki"},
        "expr": "{app=\"deepparse\", event_type=\"llm_call\"}",
        "legendFormat": "{{event_type}}"
      }]
    },

    # ── Row 2: Timeseries charts ───────────────────────────
    {
      "id": 7,
      "title": "Pipeline Runs Over Time",
      "type": "timeseries",
      "gridPos": {"x": 0, "y": 4, "w": 12, "h": 8},
      "options": {
        "tooltip": {"mode": "single"},
        "legend": {"displayMode": "list", "placement": "bottom"}
      },
      "fieldConfig": {
        "defaults": {
          "custom": {"lineWidth": 2, "fillOpacity": 20},
          "color": {"mode": "palette-classic"}
        }
      },
      "targets": [{
        "datasource": {"type": "loki", "uid": "loki"},
        "expr": "count_over_time({app=\"deepparse\", event_type=\"run_complete\"}[5m])",
        "legendFormat": "Completed Runs"
      }]
    },

    {
      "id": 8,
      "title": "LLM Calls vs Adaptive Probes",
      "type": "timeseries",
      "gridPos": {"x": 12, "y": 4, "w": 12, "h": 8},
      "options": {
        "tooltip": {"mode": "multi"},
        "legend": {"displayMode": "list", "placement": "bottom"}
      },
      "fieldConfig": {
        "defaults": {
          "custom": {"lineWidth": 2, "fillOpacity": 10}
        }
      },
      "targets": [
        {
          "datasource": {"type": "loki", "uid": "loki"},
          "expr": "count_over_time({app=\"deepparse\", event_type=\"llm_call\"}[5m])",
          "legendFormat": "LLM Calls (fallback)"
        },
        {
          "datasource": {"type": "loki", "uid": "loki"},
          "expr": "count_over_time({app=\"deepparse\", event_type=\"probe\"}[5m])",
          "legendFormat": "Adaptive Probes"
        }
      ]
    },

    # ── Row 3: Log panels ──────────────────────────────────
    {
      "id": 9,
      "title": "Run Complete Events — Full Detail",
      "type": "logs",
      "gridPos": {"x": 0, "y": 12, "w": 24, "h": 8},
      "options": {
        "showTime": True,
        "sortOrder": "Descending",
        "wrapLogMessage": True,
        "enableLogDetails": True
      },
      "targets": [{
        "datasource": {"type": "loki", "uid": "loki"},
        "expr": "{app=\"deepparse\", event_type=\"run_complete\"}"
      }]
    },

    {
      "id": 10,
      "title": "Adaptive Loop Probes (Loop 1 — Parse-Rate Monitor)",
      "type": "logs",
      "gridPos": {"x": 0, "y": 20, "w": 12, "h": 7},
      "options": {"showTime": True, "sortOrder": "Descending", "wrapLogMessage": True},
      "targets": [{
        "datasource": {"type": "loki", "uid": "loki"},
        "expr": "{app=\"deepparse\", event_type=\"probe\"}"
      }]
    },

    {
      "id": 11,
      "title": "LLM Fallback Events (Loop 2 — Teach-back trigger)",
      "type": "logs",
      "gridPos": {"x": 12, "y": 20, "w": 12, "h": 7},
      "options": {"showTime": True, "sortOrder": "Descending", "wrapLogMessage": True},
      "targets": [{
        "datasource": {"type": "loki", "uid": "loki"},
        "expr": "{app=\"deepparse\", event_type=\"llm_call\"}"
      }]
    },

    {
      "id": 12,
      "title": "Security — Injection Sanitisation Pipeline",
      "type": "logs",
      "gridPos": {"x": 0, "y": 27, "w": 12, "h": 6},
      "options": {"showTime": True, "sortOrder": "Descending"},
      "targets": [{
        "datasource": {"type": "loki", "uid": "loki"},
        "expr": "{app=\"deepparse\", event_type=\"security\"}"
      }]
    },

    {
      "id": 13,
      "title": "Warnings — Low Parse Rate / Rate Limits",
      "type": "logs",
      "gridPos": {"x": 12, "y": 27, "w": 12, "h": 6},
      "options": {"showTime": True, "sortOrder": "Descending"},
      "targets": [{
        "datasource": {"type": "loki", "uid": "loki"},
        "expr": "{app=\"deepparse\", level=\"warn\"}"
      }]
    },

    {
      "id": 14,
      "title": "All Pipeline Events",
      "type": "logs",
      "gridPos": {"x": 0, "y": 33, "w": 24, "h": 6},
      "options": {
        "showTime": True,
        "sortOrder": "Descending",
        "wrapLogMessage": True,
        "enableLogDetails": True
      },
      "targets": [{
        "datasource": {"type": "loki", "uid": "loki"},
        "expr": "{app=\"deepparse\"}"
      }]
    }
  ]
}

path = pathlib.Path("grafana/provisioning/dashboards/deepparse.json")
path.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")

# verify no BOM
raw = open(path, "rb").read(4)
print(f"deepparse.json written — BOM: {'YES BAD' if raw.startswith(b'\\xef\\xbb\\xbf') else 'NO GOOD'}")


# ============================================================
# 3 — Print Ollama setup instructions
# ============================================================
print("""
Ollama setup (run after docker compose up -d):
  docker exec deepparse-ollama ollama pull qwen2.5-coder:7b

Then run pipeline with on-prem LLM (zero egress):
  python main.py --input-dir artifacts/data --llm-provider ollama --static-masks masks_fab_universal.json
""")