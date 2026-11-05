package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func write(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadMergesConfDAndExpandsEnv(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TEST_KEY", "secret")
	write(t, filepath.Join(dir, "agent.yaml"), "# keep me\napi_key: ${TEST_KEY}\nendpoint: http://x:8080\nhostname: gw-01\ntags: [env:prod]\n")
	write(t, filepath.Join(dir, "conf.d", "nginx.yaml"), `logs:
  - type: file
    path: /var/log/nginx/*.log
    service: nginx
    log_processing_rules:
      - type: multi_line
        name: ts
        pattern: '^\d{4}-'
`)
	write(t, filepath.Join(dir, "conf.d", "remote.yaml"), "logs:\n  - type: udp\n    port: 5514\n")
	cfg, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.APIKey != "secret" || cfg.Hostname != "gw-01" || len(cfg.Sources) != 2 {
		t.Fatalf("unexpected config %+v", cfg)
	}
	if cfg.Sources[0].File != "nginx.yaml" || cfg.Sources[1].File != RemoteFile {
		t.Fatalf("sources not tagged with their file: %+v", cfg.Sources)
	}
}

func TestValidateRejectsBadSources(t *testing.T) {
	for _, s := range []Source{
		{Type: "nope"},
		{Type: "file"},
		{Type: "tcp"},
		{Type: "file", Path: "/x", Rules: []Rule{{Type: "multi_line", Pattern: "("}}},
		{Type: "file", Path: "/x", Rules: []Rule{{Type: "bogus", Pattern: "x"}}},
	} {
		if s.Validate() == nil {
			t.Errorf("expected %+v to be invalid", s)
		}
	}
}

func TestSetKeepsCommentsAndValidates(t *testing.T) {
	dir := t.TempDir()
	write(t, filepath.Join(dir, "agent.yaml"), "# keep me\nendpoint: http://a\n")
	if err := Set(dir, "endpoint", "http://b:8080"); err != nil {
		t.Fatal(err)
	}
	if err := Set(dir, "tags", "[a:1, b:2]"); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(filepath.Join(dir, "agent.yaml"))
	if !strings.Contains(string(raw), "# keep me") {
		t.Fatalf("comment lost: %s", raw)
	}
	if v, _ := Get(dir, "endpoint"); v != "http://b:8080" {
		t.Fatalf("endpoint = %q", v)
	}
	if err := Set(dir, "buffer_mb", "lots"); err == nil {
		t.Fatal("expected a type error for buffer_mb")
	}
}

func TestDefaultDirNeedsNoRoot(t *testing.T) {
	home := func() (string, error) { return "/home/u/.config", nil }
	if d := defaultDir(1000, home); d != "/home/u/.config/logpipe-agent" {
		t.Fatalf("user: %s", d)
	}
	if d := defaultDir(0, home); d != "/etc/logpipe-agent" {
		t.Fatalf("root: %s", d)
	}
}
