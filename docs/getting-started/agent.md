# Install the agent

`logpipe-agent` is a small Go collector (one static binary) that tails logs on a host and ships
them to `logpipe server` as OpenTelemetry OTLP/HTTP. It reads files (with rotation),
journald, Docker containers and syslog (TCP/UDP), joins multiline events such as stack
traces, and can drop or mask lines before they leave the host.

**Before you start**, in the logpipe UI open **Settings → API keys** and create a key with use
**Agent: ships logs** (scope `intake`). You need the key (`lp_…`) and the server URL, for example
`http://logpipe.example:8080`.

=== "Container"

    ```sh
    docker run -d --name logpipe-agent --restart unless-stopped \
      -e LOGPIPE_ENDPOINT=http://logpipe.example:8080 \
      -e LOGPIPE_API_KEY=lp_xxx \
      -e LOGPIPE_HOSTNAME=$(hostname) \
      -v /var/log:/var/log:ro \
      -v /var/run/docker.sock:/var/run/docker.sock:ro \
      -v logpipe-agent:/var/lib/logpipe-agent \
      -v $PWD/conf.d:/etc/logpipe-agent/conf.d \
      ghcr.io/niang931/logpipe-agent:latest
    ```

    - The image is built by the `Images` workflow for amd64, arm64 and arm/v7. To build it
      yourself instead: `docker build -t logpipe-agent agent/` and use `logpipe-agent` as the
      image name.
    - `/var/log` (read-only) is what it can tail; the Docker socket lets it read container logs;
      the `logpipe-agent` volume keeps file offsets and the send buffer across restarts;
      `conf.d` holds your sources (optional: sources can also be pushed from the UI).
    - The image has no `journalctl`, so for journald sources run the binary on the host.

=== "Binary (Go)"

    Needs Go 1.23+.

    ```sh
    git clone https://github.com/Niang931/Log_parser.git
    cd Log_parser/agent
    go build -o logpipe-agent .
    sudo install -m 0755 logpipe-agent /usr/local/bin/   # or keep it anywhere on your PATH
    ```

    Cross-compile for a gateway: `GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o logpipe-agent .`

    No root is needed. As a normal user the config lives in `~/.config/logpipe-agent` (offsets
    and buffer in its `run/`); as root in `/etc/logpipe-agent`. To read most of `/var/log`
    without root, add yourself to the `adm` group once: `sudo usermod -aG adm $USER`, then log out
    and back in.

## First run (binary)

```sh
logpipe-agent init --endpoint http://logpipe.example:8080 --api-key lp_xxx
logpipe-agent track '/var/log/myapp/*.log' --service myapp --source java --auto-multiline
logpipe-agent track journald:nginx.service --service nginx --source nginx
logpipe-agent check        # validates config, counts matching files, tests the server and key
logpipe-agent run          # foreground; reloads by itself when a config file changes
```

The host appears on the **Agents** page within a second of connecting.

## Basic commands

| Command | What it does |
|---|---|
| `init --endpoint URL [--api-key KEY] [--hostname H]` | write `agent.yaml` |
| `track TARGET [flags]` | add a source (`conf.d/<service>.yaml`) |
| `untrack TARGET` | remove a source you added (UI-pushed ones are removed in the UI) |
| `list` | every source and its rules |
| `check` | validate the config, the files and the connection |
| `run` | run in the foreground |
| `status` | what the running agent is doing: files, line counts, send errors, buffer size |
| `config show` | effective config, API key redacted |
| `config get KEY` / `config set KEY VALUE` | read or change `agent.yaml`, keeping comments |
| `config edit [FILE]` | edit `agent.yaml` or `conf.d/FILE.yaml` in `$EDITOR`, saved only if valid |
| `version` | the agent's version |

`TARGET` is a file glob, `journald` or `journald:UNIT`, `docker`, `tcp:PORT` or `udp:PORT`.
Useful `track` flags: `--service`, `--source`, `--tag k:v`, `--auto-multiline`,
`--multiline REGEX`, `--exclude REGEX`, `--include REGEX`, `--mask 'REGEX=>REPLACEMENT'`,
`--label key=value` (docker), `--start beginning|end`.

In the container, run the same commands through `docker exec`:

```sh
docker exec logpipe-agent logpipe-agent status
docker exec logpipe-agent logpipe-agent check
docker exec logpipe-agent logpipe-agent track '/var/log/myapp/*.log' --service myapp --source java
docker logs -f logpipe-agent
```

`track` writes into `/etc/logpipe-agent/conf.d`, so mount that directory (as above) to keep
sources across container restarts.

## Run as a service (binary)

```ini title="/etc/systemd/system/logpipe-agent.service"
[Unit]
Description=logpipe agent
After=network-online.target

[Service]
User=logpipe
Group=adm
Environment=LOGPIPE_AGENT_CONFIG=/etc/logpipe-agent
ExecStart=/usr/local/bin/logpipe-agent run
Restart=always

[Install]
WantedBy=multi-user.target
```

```sh
sudo useradd --system --no-create-home logpipe
sudo mkdir -p /etc/logpipe-agent && sudo chown logpipe /etc/logpipe-agent
sudo -u logpipe LOGPIPE_AGENT_CONFIG=/etc/logpipe-agent logpipe-agent init --endpoint http://logpipe.example:8080 --api-key lp_xxx
sudo systemctl enable --now logpipe-agent
```

## Configuration in brief

```yaml title="agent.yaml"
api_key: ${LOGPIPE_API_KEY}       # ${VAR} is read from the environment
endpoint: https://logpipe.example:8080
hostname: gw-01                   # default: the system hostname
tags: [env:prod, site:hanoi]      # added to every event
buffer_mb: 100                    # disk buffer while the server is down
remote_config: true               # let the UI push sources (conf.d/remote.yaml)
```

```yaml title="conf.d/myapp.yaml"
logs:
  - type: file
    path: /var/log/myapp/*.log
    service: myapp
    source: java
    log_processing_rules:
      - type: multi_line
        name: new_event
        pattern: '^\d{4}-\d{2}-\d{2}'
      - type: mask_sequences
        pattern: 'password=\S+'
        replace_placeholder: 'password=[REDACTED]'
```

The full reference (every rule type, Docker labels, syslog listeners, how rotation and the
buffer behave) is in
[agent/README.md](https://github.com/Niang931/Log_parser/blob/main/agent/README.md).

!!! tip "No agent? Use OpenTelemetry"
    Anything that sends OTLP/HTTP logs works too: point an OpenTelemetry Collector's `otlphttp`
    exporter at `http://<server>:8080/v1/logs` with the header `X-API-Key: lp_xxx`
    ([details](../agent-platform.md#intake-api-agents)).
