// Package ship batches events, posts them gzip-compressed to the server, and keeps batches in a
// size-capped disk buffer while the server is unreachable.
package ship

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Event struct {
	Time    string   `json:"ts"`
	Host    string   `json:"host"`
	Service string   `json:"service"`
	Source  string   `json:"source"`
	Tags    []string `json:"tags"`
	Message string   `json:"message"`
}

const (
	BatchSize  = 500
	BatchEvery = time.Second
	maxBackoff = time.Minute
)

type Client struct {
	Endpoint string
	APIKey   string
	Version  string
	HTTP     *http.Client
}

func NewClient(endpoint, key, version string) *Client {
	return &Client{Endpoint: strings.TrimRight(endpoint, "/"), APIKey: key, Version: version,
		HTTP: &http.Client{Timeout: 30 * time.Second}}
}

// Do sends body (already gzip-compressed when gz is set) and returns the response body on 2xx.
func (c *Client) Do(ctx context.Context, method, path string, body []byte, gz bool) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, method, c.Endpoint+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-API-Key", c.APIKey)
	req.Header.Set("User-Agent", "logpipe-agent/"+c.Version)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if gz {
		req.Header.Set("Content-Encoding", "gzip")
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode/100 != 2 {
		return out, &StatusError{resp.StatusCode, strings.TrimSpace(string(out))}
	}
	return out, nil
}

type StatusError struct {
	Code int
	Body string
}

func (e *StatusError) Error() string { return fmt.Sprintf("server answered %d: %s", e.Code, e.Body) }

// permanent errors will never succeed on retry: drop the batch instead of buffering it forever.
func permanent(err error) bool {
	se, ok := err.(*StatusError)
	return ok && (se.Code == 400 || se.Code == 413 || se.Code == 422)
}

type Stats struct {
	Sent        int64  `json:"sent"`
	Dropped     int64  `json:"dropped"`
	BufferBytes int64  `json:"buffer_bytes"`
	LastError   string `json:"last_error,omitempty"`
	LastSend    string `json:"last_send_at,omitempty"`
}

type Shipper struct {
	// Base carries counters over from the shipper this one replaced (config reload).
	Base     Stats
	client   *Client
	dir      string
	capBytes int64
	sent     atomic.Int64
	dropped  atomic.Int64
	mu       sync.Mutex
	lastErr  string
	lastSend time.Time
	wake     chan struct{}
}

func New(client *Client, bufferDir string, capMB int) *Shipper {
	os.MkdirAll(bufferDir, 0o755)
	return &Shipper{client: client, dir: bufferDir, capBytes: int64(capMB) << 20, wake: make(chan struct{}, 1)}
}

func (s *Shipper) Stats() Stats {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := Stats{Sent: s.Base.Sent + s.sent.Load(), Dropped: s.Base.Dropped + s.dropped.Load(),
		LastError: s.lastErr, LastSend: s.Base.LastSend}
	for _, f := range s.buffered() {
		if fi, err := os.Stat(f); err == nil {
			st.BufferBytes += fi.Size()
		}
	}
	if !s.lastSend.IsZero() {
		st.LastSend = s.lastSend.UTC().Format(time.RFC3339)
	}
	return st
}

// Run batches events from in until it is closed, then flushes what is left.
func (s *Shipper) Run(ctx context.Context, in <-chan Event) {
	go s.retryLoop(ctx)
	batch := make([]Event, 0, BatchSize)
	tick := time.NewTicker(BatchEvery)
	defer tick.Stop()
	flush := func() {
		if len(batch) > 0 {
			s.ship(ctx, batch)
			batch = batch[:0]
		}
	}
	for {
		select {
		case ev, ok := <-in:
			if !ok {
				flush()
				return
			}
			batch = append(batch, ev)
			if len(batch) >= BatchSize {
				flush()
			}
		case <-tick.C:
			flush()
		}
	}
}

func encode(batch []Event) []byte {
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	json.NewEncoder(zw).Encode(map[string]any{"events": batch})
	zw.Close()
	return buf.Bytes()
}

func (s *Shipper) ship(ctx context.Context, batch []Event) {
	body := encode(batch)
	// Older buffered batches go first; a fresh batch joins the queue rather than jumping it.
	if len(s.buffered()) == 0 {
		if err := s.post(ctx, body, len(batch)); err == nil || permanent(err) {
			return
		}
	}
	s.store(body)
}

func (s *Shipper) post(ctx context.Context, body []byte, n int) error {
	sendCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
	defer cancel()
	_, err := s.client.Do(sendCtx, "POST", "/intake/v1/logs", body, true)
	s.mu.Lock()
	defer s.mu.Unlock()
	switch {
	case err == nil:
		s.sent.Add(int64(n))
		s.lastSend, s.lastErr = time.Now(), ""
	case permanent(err):
		s.dropped.Add(int64(n))
		s.lastErr = err.Error()
	default:
		s.lastErr = err.Error()
	}
	return err
}

func (s *Shipper) buffered() []string {
	files, _ := filepath.Glob(filepath.Join(s.dir, "*.json.gz"))
	sort.Strings(files)
	return files
}

func (s *Shipper) store(body []byte) {
	name := filepath.Join(s.dir, fmt.Sprintf("%020d.json.gz", time.Now().UnixNano()))
	if err := os.WriteFile(name, body, 0o644); err != nil {
		s.mu.Lock()
		s.lastErr = "buffer: " + err.Error()
		s.mu.Unlock()
		return
	}
	// Over the cap: drop the oldest batches, the newest logs matter most when the server comes back.
	files := s.buffered()
	var total int64
	sizes := make([]int64, len(files))
	for i, f := range files {
		if fi, err := os.Stat(f); err == nil {
			sizes[i] = fi.Size()
			total += sizes[i]
		}
	}
	for i := 0; total > s.capBytes && i < len(files)-1; i++ {
		os.Remove(files[i])
		total -= sizes[i]
		s.dropped.Add(1) // counts batches, not lines: the file isn't decoded just to count
	}
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

func (s *Shipper) retryLoop(ctx context.Context) {
	backoff := time.Second
	for {
		select {
		case <-ctx.Done():
			return
		case <-s.wake:
		case <-time.After(backoff):
		}
		for _, f := range s.buffered() {
			body, err := os.ReadFile(f)
			if err != nil {
				continue
			}
			err = s.post(ctx, body, countEvents(body))
			if err != nil && !permanent(err) {
				backoff = min(backoff*2, maxBackoff)
				break
			}
			os.Remove(f)
			backoff = time.Second
		}
	}
}

func countEvents(gz []byte) int {
	zr, err := gzip.NewReader(bytes.NewReader(gz))
	if err != nil {
		return 0
	}
	var p struct{ Events []json.RawMessage }
	json.NewDecoder(zr).Decode(&p)
	return len(p.Events)
}
