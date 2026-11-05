# logpipe-agent

A small Go collector (one static binary, ~15 MB image) that tails logs on a host and ships them
to `logpipe server`. Config follows the Datadog Agent layout: one `agent.yaml` plus one file per
integration in `conf.d/`.

## Quick start (docker)

Create an API key in the UI (**API keys** page), then:

```bash
docker run -d --name logpipe-agent --restart unless-stopped \
  -e LOGPIPE_ENDPOINT=http://logpipe.example:8080 \
  -e LOGPIPE_API_KEY=lp_xxx \
  -e LOGPIPE_HOSTNAME=$(hostname) \
  -v /var/log:/var/log:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  -v logpipe-agent:/var/lib/logpipe-agent \
  ghcr.io/minhthien2103/logpipe-agent:latest
```

The agent appears on the **Agents** page as soon as it connects. Add sources there (they are pushed to the
agent as `conf.d/remote.yaml`), or mount your own `conf.d`:
`-v $PWD/conf.d:/etc/logpipe-agent/conf.d`.

## Quick start (binary)

```bash
cd agent && go build -o logpipe-agent .
./logpipe-agent init --endpoint http://logpipe.example:8080 --api-key lp_xxx
./logpipe-agent track '/var/log/myapp/*.log' --service myapp --source java --auto-multiline
./logpipe-agent check
./logpipe-agent run
```

No sudo needed. As a normal user the config lives in `~/.config/logpipe-agent` (offsets and the
send buffer in its `run/`); as root it is `/etc/logpipe-agent`. `-c DIR` (or
`$LOGPIPE_AGENT_CONFIG`) points at another directory.

The agent only needs to *read* the files it tails. Most of `/var/log` is readable by the `adm`
group, so add yourself once instead of running the agent as root:
`sudo usermod -aG adm $USER` (log out and back in). `check` names any file it can't read.

## CLI

| Command | What it does |
|---|---|
| `init --endpoint URL [--api-key KEY] [--hostname H]` | write `agent.yaml` |
| `run` | run in the foreground; reloads by itself when any config file changes |
| `track TARGET [flags]` | add a source to `conf.d/<service>.yaml` |
| `untrack TARGET` | remove it (sources pushed from the UI are removed in the UI) |
| `list` | every source and its rules |
| `config get KEY` / `config set KEY VALUE` | read or change a key in `agent.yaml`, keeping comments |
| `config edit [FILE]` | open `agent.yaml` (or `conf.d/FILE.yaml`) in `$EDITOR`; saved only if it validates |
| `config show` | effective config, API key redacted |
| `status` | what the running agent is doing: files, line counts, send errors, buffer size |
| `check` | validate config, count matching files, test the server and API key |

`TARGET` is a file glob, `journald` or `journald:UNIT`, `docker`, `tcp:PORT` or `udp:PORT`.

`track` flags: `--service`, `--source`, `--tag k:v` (repeatable), `--name FILE`,
`--multiline REGEX`, `--auto-multiline`, `--exclude REGEX`, `--include REGEX`,
`--mask 'REGEX=>REPLACEMENT'`, `--label key=value` (docker), `--start beginning|end`.

## Config reference

`agent.yaml`:

```yaml
api_key: ${LOGPIPE_API_KEY}     # ${VAR} is read from the environment
endpoint: https://logpipe.example:8080
hostname: gw-01                 # default: the system hostname
tags: [env:prod, site:hanoi]    # added to every event
run_path: /var/lib/logpipe-agent  # offsets + send buffer (default: <config dir>/run)
buffer_mb: 100                  # disk buffer while the server is down; oldest batches dropped first
remote_config: true             # let the UI push sources into conf.d/remote.yaml (default true)
auto_multi_line_detection: false  # default for sources without a multi_line rule
```

`LOGPIPE_API_KEY`, `LOGPIPE_ENDPOINT` and `LOGPIPE_HOSTNAME` override the file.

`conf.d/<name>.yaml`:

```yaml
logs:
  - type: file                  # file | journald | docker | tcp | udp
    path: /var/log/myapp/*.log  # file: glob
    service: myapp
    source: java
    tags: [team:payments]
    start_position: end         # end (default) or beginning, for files seen the first time
    log_processing_rules:
      - type: multi_line        # lines not matching start a continuation of the previous event
        name: new_event
        pattern: '^\d{4}-\d{2}-\d{2}'
      - type: exclude_at_match
        pattern: healthcheck
      - type: include_at_match
        pattern: 'ERROR|WARN'
      - type: mask_sequences
        pattern: 'password=\S+'
        replace_placeholder: 'password=[REDACTED]'
  - type: journald
    unit: nginx.service         # empty: every unit
  - type: docker
    container_labels: [logs=true]  # empty: every container
  - type: udp                   # syslog listener, RFC 3164/5424
    port: 5514
```

## How it behaves

- **Multiline**: with a `multi_line` rule, a line that does not match the pattern joins the
  previous event (Java stack traces, Python tracebacks). `auto_multi_line_detection` picks the
  pattern from the first line that starts with a known timestamp/level format. An event is sent
  when the next one starts or after 1 s of quiet, and is capped at 256 KB.
- **Rules** other than `multi_line` run in order on the whole event.
- **Files**: globs are re-checked every 250 ms; rotation (rename + new file) and copytruncate are
  both followed. Offsets are saved to `run/positions.json`, so a restart neither resends nor
  skips lines (a half-written last line is re-read, never split).
- **Shipping**: batches of up to 500 events or 1 s, gzip JSON to `POST /intake/v1/logs` with
  `X-API-Key`. Failed batches go to `run/buffer/` and are retried with backoff up to 1 min.
- **Heartbeat** every 15 s with version, sources, per-source counters and the hash of
  `remote.yaml`. When the server holds a different remote config, the agent fetches it, validates
  it, writes `conf.d/remote.yaml` and reloads. Hand-written files are never touched.
- **journald** uses `journalctl` (not in the docker image: run the binary on the host for it).
- **docker** reads the Engine API on `/var/run/docker.sock`, one stream per container.
