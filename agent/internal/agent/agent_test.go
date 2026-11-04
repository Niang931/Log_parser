package agent

import (
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jakekato/logpipe/agent/internal/ship"
	"github.com/jakekato/logpipe/agent/internal/source"
)

type fakeServer struct {
	mu     sync.Mutex
	events []ship.Event
	beats  []Status
	remote string
}

func (f *fakeServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	switch r.URL.Path {
	case "/intake/v1/logs":
		zr, _ := gzip.NewReader(r.Body)
		var p struct{ Events []ship.Event }
		json.NewDecoder(zr).Decode(&p)
		f.events = append(f.events, p.Events...)
	case "/intake/v1/heartbeat":
		var st Status
		json.NewDecoder(r.Body).Decode(&st)
		f.beats = append(f.beats, st)
		hash := ""
		if f.remote != "" {
			sum := sha256.Sum256([]byte(f.remote))
			hash = hex.EncodeToString(sum[:8])
		}
		json.NewEncoder(w).Encode(map[string]string{"remote_hash": hash})
	case "/intake/v1/config":
		json.NewEncoder(w).Encode(map[string]string{"yaml": f.remote})
	}
}

func (f *fakeServer) messages() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []string
	for _, e := range f.events {
		out = append(out, e.Message)
	}
	return out
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(25 * time.Millisecond)
	}
}

func TestAgentEndToEnd(t *testing.T) {
	HeartbeatEvery, WatchEvery, source.PollEvery = 100*time.Millisecond, 100*time.Millisecond, 20*time.Millisecond
	srv := &fakeServer{}
	ts := httptest.NewServer(srv)
	defer ts.Close()

	dir, logs := t.TempDir(), t.TempDir()
	app := filepath.Join(logs, "app.log")
	os.WriteFile(app, nil, 0o644)
	os.MkdirAll(filepath.Join(dir, "conf.d"), 0o755)
	os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte("api_key: k\nendpoint: "+ts.URL+"\nhostname: gw-1\ntags: [env:test]\n"), 0o644)
	os.WriteFile(filepath.Join(dir, "conf.d", "app.yaml"), []byte(`logs:
  - type: file
    path: `+app+`
    service: app
    source: java
    log_processing_rules:
      - type: multi_line
        pattern: '^\d{4}-'
`), 0o644)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error)
	go func() { done <- Run(ctx, dir, "test") }()
	waitFor(t, "first heartbeat", func() bool { srv.mu.Lock(); defer srv.mu.Unlock(); return len(srv.beats) > 0 })

	f, _ := os.OpenFile(app, os.O_APPEND|os.O_WRONLY, 0)
	f.WriteString("2024-01-01 00:00:00 ERROR boom\njava.lang.RuntimeException\n\tat a.b(C.java:1)\n2024-01-01 00:00:01 INFO ok\n")
	f.Close()
	waitFor(t, "aggregated trace", func() bool { return len(srv.messages()) == 2 })
	if m := srv.messages()[0]; strings.Count(m, "\n") != 2 {
		t.Fatalf("trace not aggregated: %q", m)
	}
	srv.mu.Lock()
	e := srv.events[0]
	srv.mu.Unlock()
	if e.Host != "gw-1" || e.Service != "app" || e.Source != "java" || e.Tags[0] != "env:test" {
		t.Fatalf("event metadata %+v", e)
	}

	// Push a remote config: agent writes remote.yaml and reloads with the new source.
	other := filepath.Join(logs, "other.log")
	os.WriteFile(other, nil, 0o644)
	srv.mu.Lock()
	srv.remote = "logs:\n  - type: file\n    path: " + other + "\n    service: other\n"
	srv.mu.Unlock()
	waitFor(t, "remote.yaml", func() bool {
		b, _ := os.ReadFile(filepath.Join(dir, "conf.d", "remote.yaml"))
		return strings.Contains(string(b), "other.log")
	})
	waitFor(t, "reload reports 2 sources", func() bool {
		srv.mu.Lock()
		defer srv.mu.Unlock()
		return len(srv.beats[len(srv.beats)-1].Sources) == 2
	})
	time.Sleep(100 * time.Millisecond)
	os.WriteFile(other, []byte("hello from remote\n"), 0o644)
	waitFor(t, "remote source line", func() bool { return len(srv.messages()) == 3 })
	waitFor(t, "counters kept across the reload", func() bool {
		srv.mu.Lock()
		defer srv.mu.Unlock()
		st := srv.beats[len(srv.beats)-1]
		return st.Shipper.Sent == 3 && st.Sources[0].Lines == 4
	})

	cancel()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "run", "status.json")); err != nil {
		t.Fatal("status.json not written")
	}
}
