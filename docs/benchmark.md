# Benchmark

`logpipe bench` (or `make bench`, or **Benchmark** in the UI) answers "how many logs can one
server take?". It starts a fresh server in its own process, calibrated on the synthetic IoT
fleet, and feeds it through the intake at 500, 1k, 2k, 5k, 10k, 20k, 50k and 100k lines/s,
10 s per step, until a step falls behind (processed < 95% of offered, or the buffer builds up).
A last *flood* step offers as much as the intake takes, which gives the ceiling. Each step
records offered vs processed lines/s, CPU (100% = one core), RSS, parse latency p50/p95/p99 and,
with `--store clickhouse` (a throwaway database), insert latency. Results go to
`artifacts/bench/*.json`, or for UI runs to `control_docs`, so runs can be compared. Anomaly
models are not loaded, so detector scoring is not in the numbers.

On an 8-core laptop the memory store sustained 10k lines/s, about one core, which is the GIL
bound of one Python process. ClickHouse sustained 5k with an insert p99 of about 55 ms.
