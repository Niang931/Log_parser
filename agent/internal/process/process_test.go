package process

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/jakekato/logpipe/agent/internal/config"
)

var t0 = time.Unix(0, 0)

func feed(p *Processor, lines ...string) []string {
	var evs []Event
	for i, l := range lines {
		evs = append(evs, p.Line(l, i, t0)...)
	}
	var out []string
	for _, e := range append(evs, p.Flush()...) {
		out = append(out, e.Text)
	}
	return out
}

func TestMetaIsFirstLines(t *testing.T) {
	p := New(config.Source{}, true)
	var got []any
	for i, l := range javaTrace {
		for _, e := range p.Line(l, i, t0) {
			got = append(got, e.Meta)
		}
	}
	for _, e := range p.Flush() {
		got = append(got, e.Meta)
	}
	if !reflect.DeepEqual(got, []any{0, 1, 5}) {
		t.Fatalf("meta %v", got)
	}
}

var javaTrace = []string{
	"2024-05-01 10:00:00 INFO started",
	"2024-05-01 10:00:01 ERROR boom",
	"java.lang.IllegalStateException: bad",
	"\tat com.x.Foo.bar(Foo.java:10)",
	"Caused by: java.io.IOException",
	"2024-05-01 10:00:02 INFO recovered",
}

func TestMultiLinePattern(t *testing.T) {
	p := New(config.Source{Rules: []config.Rule{{Type: "multi_line", Pattern: `^\d{4}-\d{2}-\d{2}`}}}, false)
	got := feed(p, javaTrace...)
	if len(got) != 3 || !strings.Contains(got[1], "Caused by") || strings.Count(got[1], "\n") != 3 {
		t.Fatalf("got %q", got)
	}
}

func TestAutoMultiLineFindsTimestamp(t *testing.T) {
	got := feed(New(config.Source{}, true), javaTrace...)
	if len(got) != 3 {
		t.Fatalf("auto detection gave %d events: %q", len(got), got)
	}
}

func TestNoMultiLineIsOnePerLine(t *testing.T) {
	got := feed(New(config.Source{}, false), "a", "", "b")
	if !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Fatalf("got %q", got)
	}
}

func TestLeadingContinuationIsSentAlone(t *testing.T) {
	p := New(config.Source{Rules: []config.Rule{{Type: "multi_line", Pattern: `^\d`}}}, false)
	got := feed(p, "\tat x", "1 start", "\tat y")
	if !reflect.DeepEqual(got, []string{"\tat x", "1 start\n\tat y"}) {
		t.Fatalf("got %q", got)
	}
}

func TestDueAfterTimeout(t *testing.T) {
	p := New(config.Source{}, true)
	p.Line("2024-05-01 10:00:00 a", nil, t0)
	if p.Due(t0.Add(500 * time.Millisecond)) {
		t.Fatal("flushed too early")
	}
	if !p.Due(t0.Add(FlushAfter)) {
		t.Fatal("not due after timeout")
	}
}

func TestFiltersAndMask(t *testing.T) {
	p := New(config.Source{Rules: []config.Rule{
		{Type: "exclude_at_match", Pattern: "healthcheck"},
		{Type: "include_at_match", Pattern: "GET|POST"},
		{Type: "mask_sequences", Pattern: `token=\w+`, Replacement: "token=[REDACTED]"},
	}}, false)
	got := feed(p, "GET /healthcheck", "GET /a?token=abc123", "PUT /b")
	if !reflect.DeepEqual(got, []string{"GET /a?token=[REDACTED]"}) {
		t.Fatalf("got %q", got)
	}
}

func TestEventCapped(t *testing.T) {
	p := New(config.Source{Rules: []config.Rule{{Type: "multi_line", Pattern: `^S`}}}, false)
	lines := []string{"S"}
	for i := 0; i < 5000; i++ {
		lines = append(lines, strings.Repeat("x", 100))
	}
	got := feed(p, lines...)
	if len(got) != 1 || len(got[0]) > MaxEvent {
		t.Fatalf("event not capped: %d", len(got[0]))
	}
}
