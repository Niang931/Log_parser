# Results

Latest full run of `make eval` (`artifacts/eval/REPORT.md`), offline LLM unless stated. The
nightly gate's committed baseline (`eval/baseline.json`, reduced settings) is at the end.

## Parsing: Loghub-2k

GA = grouping accuracy, PA = parsing accuracy (DeepParse's metrics). *logpipe* uses offline
masks, Drain depth 5 and similarity 0.4; *+tuned* grid-searches depth and similarity per system
on ground truth. Holdout coverage needs no labels: the share of the newest 20 % of lines a
registry mined from the older 80 % resolves without going PENDING.

| system | GA DeepParse | GA logpipe | GA +tuned | PA DeepParse | PA logpipe | PA +tuned | holdout coverage |
|---|---|---|---|---|---|---|---|
| Apache | 1.0 | 1.0 | 1.0 | 0.694 | 1.0 | 0.694 | 1.0 |
| BGL | 0.833 | 0.955 | 0.974 | 0.817 | 0.548 | 0.914 | 0.49 |
| HDFS | 0.998 | 0.158 | 0.998 | 0.223 | 0.389 | 0.223 | 0.995 |
| HPC | 0.882 | 0.888 | 0.906 | 0.882 | 0.896 | 0.882 | 0.63 |
| Hadoop | 0.685 | 0.987 | 0.961 | 0.105 | 0.394 | 0.378 | 1.0 |
| HealthApp | 0.903 | 1.0 | 0.903 | 0.766 | 0.982 | 0.764 | 0.948 |
| Linux | 0.172 | 0.178 | 0.693 | 0.156 | 0.161 | 0.219 | 0.71 |
| Mac | 0.788 | 0.799 | 0.802 | 0.494 | 0.476 | 0.450 | 0.898 |
| OpenSSH | 0.489 | 0.489 | 0.940 | 0.481 | 0.253 | 0.721 | 0.998 |
| OpenStack | 0.481 | 0.256 | 0.502 | 0.244 | 0.224 | 0.244 | 1.0 |
| Proxifier | 0.026 | 0.001 | 0.001 | 0.0 | 0.0 | 0.0 | 0.935 |
| Spark | 0.970 | 1.0 | 1.0 | 0.682 | 0.869 | 0.682 | 1.0 |
| Thunderbird | 0.637 | 0.674 | 0.677 | 0.432 | 0.338 | 0.338 | 0.998 |
| Zookeeper | 0.988 | 0.989 | 0.989 | 0.669 | 0.671 | 0.670 | 0.993 |
| **macro** | **0.703** | **0.669** | **0.810** | **0.474** | **0.514** | **0.513** | **0.900** |

**Reading it.** Default parameters lose to DeepParse on HDFS and OpenStack; tuning recovers both
and wins on Linux and OpenSSH by wide margins. Proxifier fails for every configuration here. The
tuned parameters are per system, which is what `calibrate --tune` does on a deployment's own
labelled sample.

## Alias decisions

28 hand-labelled pairs of templates: same event (should link) or opposite states (must not).

| decider | precision | recall | false merges | missed merges |
|---|---|---|---|---|
| **rules_offline** (shipping default) | **1.0** | 0.6 | **0** | 4 |
| rules + an always-yes LLM | 0.833 | 1.0 | 2 | 0 |
| token similarity only | 0.5 | 1.0 | 10 | 0 |

A missed merge leaves two ids that a person can link later; a false merge erases the
distinction anomaly detection depends on. Hence the asymmetric default.

## IoT fleet, end to end

Synthetic fleet (`logpipe gen-iot`) with a firmware rollout that changes formats mid-test and
injected anomalies of five kinds. 54,618 test events.

| variant | P | R | F1 | FP reasons | PENDING at end | resolve lag p50 | LLM calls |
|---|---|---|---|---|---|---|---|
| **full** | 0.532 | 0.583 | **0.556** | sequence 234, novelty 479 | 0 | 258 s | 13 |
| no fleet suppression | 0.234 | 0.609 | 0.338 | sequence 133, novelty 2597 | 0 | 258 s | 12 |
| mask unknowns (original LogBERT) | 0.212 | 0.349 | 0.264 | sequence 1728 | 0 | 258 s | 13 |
| no enrichment | 0.218 | 0.614 | 0.321 | sequence 234, novelty 2879 | 4975 | — | 0 |

Recall by anomaly kind (full): new event **1.0**, reorder 0.74, missing 0.65, stuck 0.53,
burst 0.26. Parse latency: p50 31.8 µs, p90 40.8, p99 68.1, max 581 µs; 416 lines/s end to end
including LogBERT training cycles and learning.

**Reading it.** Each component pays for itself: without online learning, 4,975 lines stay
PENDING and every firmware-2 event is "novel"; without fleet suppression, the rollout floods
novelty alerts; masking unknowns (the original blueprint) loses most real anomalies. Bursts are
the weak spot, plausibly because a burst of *familiar* events changes rates more than order.

## LogBERT: HDFS and BGL

Thresholds tuned on a validation split. *Oracle* is LogBERT's original protocol (tuned on the
test set), shown for comparison.

=== "BGL"

    1,146 train / 1,425 normal + 578 abnormal test sequences; vocab 148; 13 epochs in 666 s.
    Unknown templates: 0.6 % of normal test lines, 69 % of abnormal ones.

    | unknowns | novelty | P | R | F1 (val-tuned) | F1 (oracle) |
    |---|---|---|---|---|---|
    | unk | on | 0.935 | 0.971 | **0.953** | 0.954 |
    | unk | off | 0.869 | 0.869 | 0.869 | 0.878 |
    | mask | on | 0.935 | 0.971 | 0.953 | 0.954 |
    | mask | off | 0.691 | 0.282 | 0.401 | 0.401 |
    | drop | off | 0.678 | 0.270 | 0.386 | 0.386 |

=== "HDFS"

    8,000 train / 10,000 normal + 1,001 abnormal test sessions; vocab 20; 30 epochs in 189 s.

    | unknowns | novelty | P | R | F1 (val-tuned) | F1 (oracle) |
    |---|---|---|---|---|---|
    | unk | on | 0.970 | 0.683 | **0.802** | 0.802 |
    | unk | off | 0.963 | 0.514 | 0.670 | 0.670 |
    | mask | off | 0.963 | 0.513 | 0.669 | 0.669 |
    | drop | off | 0.957 | 0.443 | 0.605 | 0.605 |

**Reading it.** On BGL most anomalies contain templates never seen in training, which is
exactly what masking hides (F1 0.40) and the novelty signal catches (0.95).

## RCA on injected faults (synthetic)

`python -m logpipe.eval.rca_eval`: a web shop's normal traffic for 2 h, then 15 fault modes
(SIGTERM, OOM, segfault, disk full, DB port blocked, pool exhausted, gateway timeout, DNS, TLS
expired, config reload, dependency 503, GC thrash, lock contention, clock skew, API hang) plus
a bad host, a cross-service trace hop and 2 no-fault controls, 19 in all. Each asks an on-call
question.

| variant | evidence recall | answer correct |
|---|---|---|
| naive (last 200 lines) | 0.733 | — |
| **full, rules only** | **1.0** | **0.947** (18/19) |
| full + Gemini (subset re-run) | 1.0 | `trace_hop` and `bad_host` both correct |

The one rules-only miss is `trace_hop` (the cause is a Postgres line linked by request id); the
LLM's typed `trace` move fixes it. **Caveat:** the heuristics were tuned on these scenarios, so
this is a fit, not a held-out score. That's why the RCAEval results below matter.

## RCA on RCAEval (real testbeds)

[Full page →](rcaeval.md). RE3-OB, 30 code-level faults on Online Boutique:

| setup | AC@1 | AC@3 | Avg@5 |
|---|---|---|---|
| rules only | 0.40 | 0.40 | 0.56 |
| rules + Gemini (web) | 0.70 | 0.77 | 0.81 |
| Claude Code over MCP, with `ask` | 0.87 | 0.87 | 0.88 |
| Claude Code over MCP, raw tools | 0.90 | 0.90 | 0.90 |

## Nightly gate baseline

`eval/baseline.json`: reduced settings (`--no-tune`; IoT with 3 epochs, 3 devices, 2 h) so CI
fits in minutes. Not comparable with the full-run numbers above.

| check | baseline | tolerance |
|---|---|---|
| parsing mean GA (15 systems incl. IoT) | 0.8037 | −0.02 |
| alias false merges (offline) | 0 | hard limit |
| IoT F1 | 0.5147 | −0.05 |
