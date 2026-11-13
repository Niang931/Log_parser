# Anomaly detection (LogBERT)

logpipe models **sequences of template ids** per device or stream with LogBERT (Guo et al.,
IJCNN 2021), vendored and adapted in `logpipe/anomaly/logbert/`. Keyword rules catch
known-bad lines; LogBERT catches *unusual order, missing steps, bursts and stuck loops* in
lines that each look normal.

## Model

| Part | Default | Code |
|---|---|---|
| Vocabulary | template ids → keys, plus `[PAD]`, `[SOS]`, `[MASK]`, `[UNK]` | `anomaly/vocab.py` |
| Encoder | BERT, hidden 64, 2 layers, 4 heads, max length 128, dropout 0.1 | `logbert/model.py` |
| Objectives | masked log-key prediction + hypersphere (`[SOS]` embedding near the centre) | `logbert/trainer.py` |
| Windows | 20 events, stride 5, minimum length 5 | `detector.py:DetectorConfig` |

Ids are **canonical** (aliases resolved), so a firmware rewording that was linked to the old
template is the same token to the model.

## Scoring a window

```mermaid
flowchart LR
    W[window of 20 template ids] --> M[mask a share of keys]
    M --> B[LogBERT predicts each masked key]
    B --> H{true key in top-g?}
    H --> MR[miss_ratio = misses / masked]
    W --> NR[novelty_ratio = never-seen templates / length]
    B --> DS[distance of SOS embedding<br/>from the centre, optional]
    MR & NR & DS --> T{over thresholds?}
    T -->|yes| AL[AnomalyAlert with reasons:<br/>sequence / novelty / distance]
```

- **`miss_ratio`**: LogBERT's own criterion. Most masked keys missing from the top-g predictions
  means the sequence isn't one the model knows.
- **`novelty_ratio`**: a separate signal added here. A never-seen event type is often the
  clearest anomaly on a device; the original blueprint masked unknowns, which hides it. The
  ablation (`unknown_mode = unk | mask | drop`) shows the effect: on BGL, F1 with novelty is
  **0.95**, and without it 0.87 (`unk`), 0.40 (`mask`) or 0.39 (`drop`).
- **Thresholds** (`scorer.py:tune_thresholds`): grid search for F1 on a *validation* split. The
  original tuned on the test set, which overstates F1; the eval reports both. Without labels
  (the server), a stream's threshold is the 99th percentile of the miss ratio on held-out normal
  windows (at least 0.3), and novelty is set to `inf`: unknown templates become a
  *retrain item* in Maintenance, not an alert.

## The streaming detector

`anomaly/detector.py`:

```mermaid
flowchart LR
    E[event: device, time, template] --> RB[per-device heap<br/>reorder buffer, lateness 2 s]
    RB --> SW[per-device sliding window]
    SW -->|every 5 events| QW[queued windows]
    QW -->|batch of 64 across devices| LB[one LogBERT forward pass]
    LB --> FS{fleet suppression}
    FS -->|novel template seen on ≥ 2 devices<br/>for ≥ 10 minutes| SUP[treated as familiar]
    FS --> AL[alert]
```

- **Reorder buffer.** Device clocks drift and MQTT reorders across reconnects, so events wait
  until older than (newest − lateness).
- **Batching.** One forward pass per 64 windows across devices, not one per event.
- **Fleet suppression.** A template that appears on many devices over a while is a rollout, not
  an anomaly. On the IoT fleet this cuts novelty false positives from 2,597 to 479 and lifts F1
  from 0.34 to 0.56 ([results](../benchmarks/results.md#iot-fleet-end-to-end)).

## In the server: per-stream models

A **stream** is a named list of `host` / `service` / `source` globs. Each trains its own
LogBERT on capped per-source sequences. A model needs 8 windows (20 consecutive parsed lines
from one source); with less, the stream waits with a have/need meter. Retraining runs on
request, or when a model is older than `retrain_days` (7) **and** has templates it hasn't seen.

**Drift.** Training keeps a histogram of how much of each held-out window the model missed;
live windows feed a rolling histogram per stream (the last 2,000). The PSI between the two
says how far traffic has moved: < 0.1 stable, 0.1–0.25 moderate, > 0.25 retrain.

## Where its scores go

- `anomaly_alerts` (ClickHouse) or the store's alert list: windows over threshold, with
  `miss_ratio`, `novelty_ratio`, reasons and template ids.
- **Incidents**: alerts on one service closer than 5 minutes are grouped into one incident.
- **RCA**: `scorer.explain(window)` re-reads the anchor service's last lines and marks each
  `expected | unexpected | unseen`; the share of surprises becomes a candidate's anomaly
  factor ([Ask & RCA](ask-rca.md)).
