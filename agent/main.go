// logpipe-agent: tails logs on a host and ships them to a logpipe server.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/jakekato/logpipe/agent/internal/agent"
	"github.com/jakekato/logpipe/agent/internal/config"
	"github.com/jakekato/logpipe/agent/internal/ship"
	"github.com/jakekato/logpipe/agent/internal/source"
	"gopkg.in/yaml.v3"
)

var version = "dev"

const usage = `logpipe-agent: ship logs from this host to a logpipe server

usage: logpipe-agent [-c DIR] <command> [args]

commands:
  init      --endpoint URL [--api-key KEY]   write agent.yaml
  run                                         run in the foreground (reloads when config changes)
  track     TARGET [flags]                    start collecting a file glob, journald unit, docker or syslog port
  untrack   TARGET                            stop collecting it
  list                                        show every tracked source
  config    get KEY | set KEY VALUE | edit [FILE] | show | path
  status                                      what the running agent is doing
  check                                       validate config and test the server connection
  version

-c defaults to $LOGPIPE_AGENT_CONFIG or /etc/logpipe-agent.
`

func main() {
	log.SetFlags(log.LstdFlags | log.Lmsgprefix)
	log.SetPrefix("[agent] ")
	global := flag.NewFlagSet("logpipe-agent", flag.ExitOnError)
	dir := global.String("c", config.DefaultDir(), "config directory")
	global.Usage = func() { fmt.Fprint(os.Stderr, usage) }
	global.Parse(os.Args[1:])
	args := global.Args()
	if len(args) == 0 {
		global.Usage()
		os.Exit(2)
	}
	if err := dispatch(*dir, args[0], args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func dispatch(dir, cmd string, args []string, out io.Writer) error {
	switch cmd {
	case "init":
		return cmdInit(dir, args, out)
	case "run":
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer stop()
		return agent.Run(ctx, dir, version)
	case "track":
		return cmdTrack(dir, args, out)
	case "untrack":
		return cmdUntrack(dir, args, out)
	case "list":
		return cmdList(dir, out)
	case "config":
		return cmdConfig(dir, args, out)
	case "status":
		return cmdStatus(dir, out)
	case "check":
		return cmdCheck(dir, out)
	case "version":
		fmt.Fprintln(out, version)
		return nil
	}
	return fmt.Errorf("unknown command %q (see logpipe-agent -h)", cmd)
}

func cmdInit(dir string, args []string, out io.Writer) error {
	fs := flag.NewFlagSet("init", flag.ContinueOnError)
	endpoint := fs.String("endpoint", "", "server URL, e.g. http://logpipe:8080")
	key := fs.String("api-key", "${LOGPIPE_API_KEY}", "API key (default reads $LOGPIPE_API_KEY at start)")
	host := fs.String("hostname", "", "name shown in the UI (default: system hostname)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *endpoint == "" {
		return errors.New("--endpoint is required")
	}
	path := filepath.Join(dir, "agent.yaml")
	if _, err := os.Stat(path); err == nil {
		return fmt.Errorf("%s already exists; use `config set`", path)
	}
	body := fmt.Sprintf("# logpipe-agent main config. Sources live in conf.d/*.yaml.\napi_key: %q\nendpoint: %s\n", *key, *endpoint)
	if *host != "" {
		body += "hostname: " + *host + "\n"
	}
	body += "tags: []\n# auto_multi_line_detection: true\n# remote_config: true   # let the UI push sources into conf.d/remote.yaml\n"
	if err := os.MkdirAll(filepath.Join(dir, "conf.d"), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		return err
	}
	fmt.Fprintln(out, "wrote", path)
	return nil
}

type multiFlag []string

func (m *multiFlag) String() string     { return strings.Join(*m, ",") }
func (m *multiFlag) Set(v string) error { *m = append(*m, v); return nil }

// sourceFromTarget reads TARGET as a file glob, journald[:unit], docker, or tcp:PORT / udp:PORT.
func sourceFromTarget(target string) config.Source {
	switch {
	case target == "journald" || strings.HasPrefix(target, "journald:"):
		return config.Source{Type: "journald", Unit: strings.TrimPrefix(strings.TrimPrefix(target, "journald"), ":")}
	case target == "docker":
		return config.Source{Type: "docker"}
	case strings.HasPrefix(target, "tcp:") || strings.HasPrefix(target, "udp:"):
		var port int
		fmt.Sscan(target[4:], &port)
		return config.Source{Type: target[:3], Port: port}
	}
	if abs, err := filepath.Abs(target); err == nil {
		target = abs
	}
	return config.Source{Type: "file", Path: target}
}

func cmdTrack(dir string, args []string, out io.Writer) error {
	fs := flag.NewFlagSet("track", flag.ContinueOnError)
	service := fs.String("service", "", "service name")
	src := fs.String("source", "", "source (technology) name, e.g. nginx, java")
	file := fs.String("name", "", "conf.d file name (default: the service, else \"custom\")")
	multiline := fs.String("multiline", "", "regex every new event starts with (e.g. '^\\d{4}-\\d{2}-\\d{2}')")
	auto := fs.Bool("auto-multiline", false, "detect the event start automatically")
	start := fs.String("start", "", "end (default) or beginning: where to start a file seen for the first time")
	var tags, exclude, include, mask, labels multiFlag
	fs.Var(&tags, "tag", "tag key:value (repeatable)")
	fs.Var(&exclude, "exclude", "drop events matching regex (repeatable)")
	fs.Var(&include, "include", "keep only events matching regex (repeatable)")
	fs.Var(&mask, "mask", "REGEX=>REPLACEMENT redaction (repeatable)")
	fs.Var(&labels, "label", "docker: only containers with this label key=value (repeatable)")
	target, rest := splitTarget(args)
	if err := fs.Parse(rest); err != nil {
		return err
	}
	if target == "" {
		target = fs.Arg(0) // flags first, target last
	}
	if target == "" {
		return errors.New("usage: track TARGET [flags]  (TARGET: file glob, journald[:unit], docker, tcp:PORT, udp:PORT)")
	}
	s := sourceFromTarget(target)
	s.Service, s.Src, s.Tags, s.Labels, s.StartAt = *service, *src, tags, labels, *start
	if *multiline != "" {
		s.Rules = append(s.Rules, config.Rule{Type: "multi_line", Name: "new_event", Pattern: *multiline})
	}
	if *auto {
		t := true
		s.AutoMultiLine = &t
	}
	for _, p := range exclude {
		s.Rules = append(s.Rules, config.Rule{Type: "exclude_at_match", Pattern: p})
	}
	for _, p := range include {
		s.Rules = append(s.Rules, config.Rule{Type: "include_at_match", Pattern: p})
	}
	for _, m := range mask {
		re, repl, ok := strings.Cut(m, "=>")
		if !ok {
			return fmt.Errorf("--mask wants REGEX=>REPLACEMENT, got %q", m)
		}
		s.Rules = append(s.Rules, config.Rule{Type: "mask_sequences", Pattern: re, Replacement: repl})
	}
	if err := s.Validate(); err != nil {
		return err
	}
	if cfg, err := config.Load(dir); err == nil {
		for _, existing := range cfg.Sources {
			if existing.ID() == s.ID() {
				return fmt.Errorf("%s is already tracked in conf.d/%s", source.Target(s), existing.File)
			}
		}
	}
	name := *file
	if name == "" {
		name = *service
	}
	if name == "" {
		name = "custom"
	}
	name = strings.TrimSuffix(name, ".yaml") + ".yaml"
	if name == config.RemoteFile {
		return errors.New("remote.yaml is managed by the server; pick another --name")
	}
	path := filepath.Join(dir, "conf.d", name)
	integ, err := config.ReadIntegration(path)
	if err != nil {
		return err
	}
	integ.Logs = append(integ.Logs, s)
	if err := config.WriteIntegration(path, integ); err != nil {
		return err
	}
	fmt.Fprintf(out, "tracking %s (%s) in %s; a running agent picks it up within seconds\n", source.Target(s), s.Type, path)
	return nil
}

// splitTarget accepts the target before the flags (`track /var/log/x.log --service x`).
func splitTarget(args []string) (string, []string) {
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		return args[0], args[1:]
	}
	return "", args
}

func cmdUntrack(dir string, args []string, out io.Writer) error {
	if len(args) != 1 {
		return errors.New("usage: untrack TARGET")
	}
	want := sourceFromTarget(args[0]).ID()
	files, _ := filepath.Glob(filepath.Join(dir, "conf.d", "*.yaml"))
	for _, f := range files {
		integ, err := config.ReadIntegration(f)
		if err != nil {
			return err
		}
		kept := integ.Logs[:0]
		for _, s := range integ.Logs {
			if s.ID() != want {
				kept = append(kept, s)
			}
		}
		if len(kept) == len(integ.Logs) {
			continue
		}
		if filepath.Base(f) == config.RemoteFile {
			return errors.New("that source was pushed from the server; remove it in the UI")
		}
		integ.Logs = kept
		if err := config.WriteIntegration(f, integ); err != nil {
			return err
		}
		fmt.Fprintf(out, "untracked %s (%s)\n", args[0], f)
		return nil
	}
	return fmt.Errorf("%s is not tracked", args[0])
}

func cmdList(dir string, out io.Writer) error {
	cfg, err := config.Load(dir)
	if err != nil {
		return err
	}
	w := tabwriter.NewWriter(out, 0, 2, 2, ' ', 0)
	fmt.Fprintln(w, "FILE\tTYPE\tTARGET\tSERVICE\tRULES")
	for _, s := range cfg.Sources {
		var rules []string
		for _, r := range s.Rules {
			rules = append(rules, r.Type)
		}
		if s.AutoMultiLine != nil && *s.AutoMultiLine || s.AutoMultiLine == nil && cfg.AutoMultiLine {
			rules = append(rules, "auto_multi_line")
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", s.File, s.Type, source.Target(s), s.Service, strings.Join(rules, ","))
	}
	return w.Flush()
}

func cmdConfig(dir string, args []string, out io.Writer) error {
	if len(args) == 0 {
		return errors.New("usage: config get KEY | set KEY VALUE | edit [FILE] | show | path")
	}
	switch args[0] {
	case "get":
		if len(args) != 2 {
			return errors.New("usage: config get KEY")
		}
		v, err := config.Get(dir, args[1])
		if err == nil {
			fmt.Fprintln(out, v)
		}
		return err
	case "set":
		if len(args) != 3 {
			return errors.New("usage: config set KEY VALUE")
		}
		if err := config.Set(dir, args[1], args[2]); err != nil {
			return err
		}
		fmt.Fprintf(out, "%s updated\n", args[1])
		return nil
	case "edit":
		path := filepath.Join(dir, "agent.yaml")
		if len(args) > 1 {
			path = filepath.Join(dir, "conf.d", strings.TrimSuffix(args[1], ".yaml")+".yaml")
		}
		return edit(path, out)
	case "show":
		cfg, err := config.Load(dir)
		if err != nil {
			return err
		}
		a := cfg.Agent
		a.APIKey = redact(a.APIKey)
		b, _ := yaml.Marshal(map[string]any{"agent": a, "sources": cfg.Sources})
		_, err = out.Write(b)
		return err
	case "path":
		fmt.Fprintln(out, dir)
		return nil
	}
	return fmt.Errorf("unknown config command %q", args[0])
}

func redact(k string) string {
	if len(k) <= 4 {
		return strings.Repeat("*", len(k))
	}
	return strings.Repeat("*", len(k)-4) + k[len(k)-4:]
}

// edit opens a copy in $EDITOR and only replaces the file once the copy validates.
func edit(path string, out io.Writer) error {
	if filepath.Base(path) == config.RemoteFile {
		return errors.New("remote.yaml is managed by the server; edit it in the UI")
	}
	editor := os.Getenv("VISUAL")
	if editor == "" {
		editor = os.Getenv("EDITOR")
	}
	if editor == "" {
		editor = "vi"
	}
	orig, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if len(orig) == 0 && filepath.Base(path) != "agent.yaml" {
		orig = []byte("logs:\n  - type: file\n    path: /var/log/example.log\n    service: example\n")
	}
	tmp, err := os.CreateTemp("", "logpipe-agent-*.yaml")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	tmp.Write(orig)
	tmp.Close()
	for {
		cmd := exec.Command("sh", "-c", editor+` "$0"`, tmp.Name())
		cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("editor: %w", err)
		}
		edited, _ := os.ReadFile(tmp.Name())
		if string(edited) == string(orig) {
			fmt.Fprintln(out, "no changes")
			return nil
		}
		if err := config.CheckFile(path, edited); err != nil {
			fmt.Fprintf(out, "invalid: %v\nreopen the editor? [Y/n] ", err)
			var ans string
			fmt.Scanln(&ans)
			if strings.HasPrefix(strings.ToLower(ans), "n") {
				return errors.New("changes discarded")
			}
			continue
		}
		os.MkdirAll(filepath.Dir(path), 0o755)
		if err := os.WriteFile(path, edited, 0o644); err != nil {
			return err
		}
		fmt.Fprintln(out, "saved", path)
		return nil
	}
}

func cmdStatus(dir string, out io.Writer) error {
	cfg, err := config.Load(dir)
	if err != nil {
		return err
	}
	b, err := os.ReadFile(agent.StatusPath(cfg))
	if err != nil {
		return errors.New("no status yet: is `logpipe-agent run` running with this config?")
	}
	var st agent.Status
	if err := json.Unmarshal(b, &st); err != nil {
		return err
	}
	if t, err := time.Parse(time.RFC3339, st.Updated); err == nil && time.Since(t) > 30*time.Second {
		fmt.Fprintf(out, "WARNING: status is %s old, the agent may be stopped\n", time.Since(t).Round(time.Second))
	}
	fmt.Fprintf(out, "agent %s on %s -> %s\nstarted %s, last heartbeat %s\n", st.Version, st.Hostname, st.Endpoint, st.StartedAt, orDash(st.Heartbeat))
	fmt.Fprintf(out, "shipped %d events, dropped %d, buffered %d bytes, last send %s\n", st.Shipper.Sent, st.Shipper.Dropped, st.Shipper.BufferBytes, orDash(st.Shipper.LastSend))
	if st.Shipper.LastError != "" {
		fmt.Fprintln(out, "last send error:", st.Shipper.LastError)
	}
	if st.ConfigError != "" {
		fmt.Fprintln(out, "config error (running previous config):", st.ConfigError)
	}
	w := tabwriter.NewWriter(out, 0, 2, 2, ' ', 0)
	fmt.Fprintln(w, "\nTYPE\tTARGET\tLINES\tBYTES\tLAST LINE\tERROR")
	for _, s := range st.Sources {
		fmt.Fprintf(w, "%s\t%s\t%d\t%d\t%s\t%s\n", s.Type, s.Target, s.Lines, s.Bytes, orDash(s.LastLine), s.Error)
		for _, f := range s.Files {
			fmt.Fprintf(w, "\t  %s\t\t\t\t\n", f)
		}
	}
	return w.Flush()
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func cmdCheck(dir string, out io.Writer) error {
	cfg, err := config.Load(dir)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "config OK: %d sources\n", len(cfg.Sources))
	for _, s := range cfg.Sources {
		switch s.Type {
		case "file":
			m, _ := filepath.Glob(s.Path)
			fmt.Fprintf(out, "  %s: %d files match\n", s.Path, len(m))
		case "journald":
			if _, err := exec.LookPath("journalctl"); err != nil {
				fmt.Fprintln(out, "  journald: journalctl not found")
			}
		case "docker":
			if _, err := os.Stat(source.DockerSocket()); err != nil {
				fmt.Fprintf(out, "  docker: %v\n", err)
			}
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := ship.NewClient(cfg.Endpoint, cfg.APIKey, version).Do(ctx, "GET", "/intake/v1/ping", nil, false); err != nil {
		return fmt.Errorf("server %s: %w", cfg.Endpoint, err)
	}
	fmt.Fprintf(out, "server %s accepted the API key\n", cfg.Endpoint)
	return nil
}
