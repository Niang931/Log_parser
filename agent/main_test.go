package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jakekato/logpipe/agent/internal/config"
)

func run(t *testing.T, dir string, args ...string) string {
	t.Helper()
	var out bytes.Buffer
	if err := dispatch(dir, args[0], args[1:], &out); err != nil {
		t.Fatalf("%v: %v", args, err)
	}
	return out.String()
}

func TestTrackListUntrack(t *testing.T) {
	dir := t.TempDir()
	run(t, dir, "init", "--endpoint", "http://srv:8080", "--api-key", "k")
	run(t, dir, "track", "/var/log/app/*.log", "--service", "app", "--multiline", `^\d{4}`, "--exclude", "health", "--mask", `token=\w+=>token=***`, "--tag", "team:a")
	run(t, dir, "track", "--service", "app", "journald:app.service")
	run(t, dir, "track", "udp:5514", "--name", "net")

	cfg, err := config.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Sources) != 3 {
		t.Fatalf("sources %+v", cfg.Sources)
	}
	s := cfg.Sources[0]
	if s.File != "app.yaml" || len(s.Rules) != 3 || s.Rules[2].Replacement != "token=***" || s.Tags[0] != "team:a" {
		t.Fatalf("file source %+v", s)
	}
	if cfg.Sources[1].Unit != "app.service" || cfg.Sources[2].File != "net.yaml" {
		t.Fatalf("got %+v", cfg.Sources)
	}

	var out bytes.Buffer
	if err := dispatch(dir, "track", []string{"/var/log/app/*.log"}, &out); err == nil {
		t.Fatal("duplicate track should fail")
	}
	if list := run(t, dir, "list"); !strings.Contains(list, "/var/log/app/*.log") || !strings.Contains(list, "multi_line") {
		t.Fatalf("list:\n%s", list)
	}
	run(t, dir, "untrack", "udp:5514")
	if _, err := os.Stat(filepath.Join(dir, "conf.d", "net.yaml")); !os.IsNotExist(err) {
		t.Fatal("empty conf.d file should be removed")
	}
}

func TestConfigSetGetShow(t *testing.T) {
	dir := t.TempDir()
	run(t, dir, "init", "--endpoint", "http://a", "--api-key", "secretkey")
	run(t, dir, "config", "set", "endpoint", "http://b")
	if got := strings.TrimSpace(run(t, dir, "config", "get", "endpoint")); got != "http://b" {
		t.Fatalf("get = %q", got)
	}
	if show := run(t, dir, "config", "show"); strings.Contains(show, "secretkey") {
		t.Fatal("show leaks the api key")
	}
}

func TestUntrackRefusesRemote(t *testing.T) {
	dir := t.TempDir()
	run(t, dir, "init", "--endpoint", "http://a")
	os.WriteFile(filepath.Join(dir, "conf.d", "remote.yaml"), []byte("logs:\n  - type: tcp\n    port: 601\n"), 0o644)
	var out bytes.Buffer
	if err := dispatch(dir, "untrack", []string{"tcp:601"}, &out); err == nil || !strings.Contains(err.Error(), "server") {
		t.Fatalf("got %v", err)
	}
}
