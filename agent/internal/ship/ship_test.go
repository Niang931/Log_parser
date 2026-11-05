package ship

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type server struct {
	mu     sync.Mutex
	got    []string
	paths  []string
	down   atomic.Bool
	key    string
	lastRq otlpBody
}

// otlpBody is the part of an OTLP/HTTP JSON logs request the tests read.
type otlpBody struct {
	ResourceLogs []struct {
		Resource struct {
			Attributes []struct {
				Key   string
				Value struct{ StringValue string }
			}
		}
		ScopeLogs []struct {
			LogRecords []struct {
				ObservedTimeUnixNano string
				Body                 struct{ StringValue string }
				Attributes           []struct {
					Key   string
					Value struct{ StringValue string }
				}
			}
		}
	}
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if s.down.Load() {
		w.WriteHeader(503)
		return
	}
	if r.Header.Get("X-API-Key") != s.key {
		w.WriteHeader(401)
		return
	}
	zr, _ := gzip.NewReader(r.Body)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.paths = append(s.paths, r.URL.Path)
	if r.URL.Path == LegacyPath {
		var p struct{ Events []struct{ Message string } }
		json.NewDecoder(zr).Decode(&p)
		for _, e := range p.Events {
			s.got = append(s.got, e.Message)
		}
		return
	}
	var p otlpBody
	json.NewDecoder(zr).Decode(&p)
	s.lastRq = p
	for _, rl := range p.ResourceLogs {
		for _, sl := range rl.ScopeLogs {
			for _, lr := range sl.LogRecords {
				s.got = append(s.got, lr.Body.StringValue)
			}
		}
	}
}

func (s *server) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.got)
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal("timed out")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestBuffersWhileDownThenDelivers(t *testing.T) {
	srv := &server{key: "k"}
	srv.down.Store(true)
	ts := httptest.NewServer(srv)
	defer ts.Close()
	sh := New(NewClient(ts.URL, "k", "test"), t.TempDir(), 10)
	in := make(chan Event)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go sh.Run(ctx, in)
	for i := 0; i < 3; i++ {
		in <- Event{Message: "m"}
	}
	waitFor(t, func() bool { return sh.Stats().BufferBytes > 0 })
	srv.down.Store(false)
	in <- Event{Message: "m"}
	waitFor(t, func() bool { return srv.count() == 4 && sh.Stats().BufferBytes == 0 })
	if st := sh.Stats(); st.Sent != 4 || st.LastError != "" {
		t.Fatalf("stats %+v", st)
	}
}

func TestShipsOTLPWithSemanticConventions(t *testing.T) {
	srv := &server{key: "k"}
	ts := httptest.NewServer(srv)
	defer ts.Close()
	sh := New(NewClient(ts.URL, "k", "1.2"), t.TempDir(), 10)
	at := time.Date(2026, 1, 2, 3, 4, 5, 6, time.FixedZone("", 8*3600))
	sh.ship(context.Background(), []Event{
		{Time: at, Host: "h1", Service: "api", Source: "java", Path: "/var/log/app.log", Tags: []string{"env:prod"}, Message: "a"},
		{Time: at, Host: "h1", Service: "api", Source: "java", Message: "b"},
		{Time: at, Host: "h1", Service: "db", Source: "pg", Message: "c"},
	})
	srv.mu.Lock()
	defer srv.mu.Unlock()
	if srv.paths[0] != "/v1/logs" || len(srv.lastRq.ResourceLogs) != 2 {
		t.Fatalf("paths %v, resources %d", srv.paths, len(srv.lastRq.ResourceLogs))
	}
	res := map[string]string{}
	for _, a := range srv.lastRq.ResourceLogs[0].Resource.Attributes {
		res[a.Key] = a.Value.StringValue
	}
	if res["host.name"] != "h1" || res["service.name"] != "api" || res["logpipe.source"] != "java" || res["logpipe.host.utc_offset"] != "+08:00" {
		t.Fatalf("resource %v", res)
	}
	lr := srv.lastRq.ResourceLogs[0].ScopeLogs[0].LogRecords
	if len(lr) != 2 || lr[0].ObservedTimeUnixNano != strconv.FormatInt(at.UnixNano(), 10) || lr[0].Attributes[0].Key != "log.file.path" {
		t.Fatalf("records %+v", lr)
	}
}

func TestLegacyBufferedBatchesStillGoToTheOldEndpoint(t *testing.T) {
	srv := &server{key: "k"}
	ts := httptest.NewServer(srv)
	defer ts.Close()
	dir := t.TempDir()
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	zw.Write([]byte(`{"events":[{"message":"old"}]}`))
	zw.Close()
	os.WriteFile(filepath.Join(dir, "00000000000000000001.json.gz"), buf.Bytes(), 0o644)
	sh := New(NewClient(ts.URL, "k", "t"), dir, 10)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go sh.retryLoop(ctx)
	waitFor(t, func() bool { return srv.count() == 1 })
	srv.mu.Lock()
	defer srv.mu.Unlock()
	if srv.paths[0] != LegacyPath || srv.got[0] != "old" {
		t.Fatalf("%v %v", srv.paths, srv.got)
	}
}

func TestBufferCapDropsOldest(t *testing.T) {
	sh := New(NewClient("http://127.0.0.1:1", "k", "t"), t.TempDir(), 0)
	sh.store([]byte("aaaa"))
	sh.store([]byte("bbbb"))
	if n := len(sh.buffered()); n != 1 {
		t.Fatalf("kept %d batches over a zero cap", n)
	}
}
