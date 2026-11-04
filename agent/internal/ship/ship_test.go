package ship

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type server struct {
	mu   sync.Mutex
	got  []string
	down atomic.Bool
	key  string
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
	var p struct{ Events []Event }
	json.NewDecoder(zr).Decode(&p)
	s.mu.Lock()
	for _, e := range p.Events {
		s.got = append(s.got, e.Message)
	}
	s.mu.Unlock()
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

func TestBufferCapDropsOldest(t *testing.T) {
	sh := New(NewClient("http://127.0.0.1:1", "k", "t"), t.TempDir(), 0)
	sh.store([]byte("aaaa"))
	sh.store([]byte("bbbb"))
	if n := len(sh.buffered()); n != 1 {
		t.Fatalf("kept %d batches over a zero cap", n)
	}
}
