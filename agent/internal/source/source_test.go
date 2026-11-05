package source

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/jakekato/logpipe/agent/internal/config"
)

func init() { PollEvery = 10 * time.Millisecond }

func collect(t *testing.T, out <-chan Line, n int) []string {
	t.Helper()
	var got []string
	timeout := time.After(3 * time.Second)
	for len(got) < n {
		select {
		case l := <-out:
			got = append(got, l.Text)
		case <-timeout:
			t.Fatalf("timed out after %q", got)
		}
	}
	return got
}

func appendFile(t *testing.T, path, s string) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString(s)
	f.Close()
}

func TestTailRotationTruncationAndResume(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "app.log")
	appendFile(t, path, "old line before start\n")
	pos := LoadPositions(filepath.Join(dir, "positions.json"))
	src := config.Source{Type: "file", Path: filepath.Join(dir, "*.log")}
	ctx, cancel := context.WithCancel(context.Background())
	out := make(chan Line, 100)
	done := make(chan struct{})
	go func() { TailFiles(ctx, src, pos, &Tracker{}, out); close(done) }()

	time.Sleep(50 * time.Millisecond)
	appendFile(t, path, "a\npart")
	if got := collect(t, out, 1); got[0] != "a" {
		t.Fatalf("start_position end not honoured: %q", got)
	}
	appendFile(t, path, "ial\n")
	if got := collect(t, out, 1); got[0] != "partial" {
		t.Fatalf("partial line split: %q", got)
	}
	// rotate: rename + new file
	os.Rename(path, path+".1")
	appendFile(t, path, "after rotate\n")
	if got := collect(t, out, 1); got[0] != "after rotate" {
		t.Fatalf("rotation: %q", got)
	}
	// truncate in place
	os.Truncate(path, 0)
	time.Sleep(50 * time.Millisecond)
	appendFile(t, path, "x\n")
	if got := collect(t, out, 1); got[0] != "x" {
		t.Fatalf("truncation: %q", got)
	}
	cancel()
	<-done
	pos.Save()

	// restart: resumes at the saved offset, no resend
	appendFile(t, path, "while down\n")
	pos2 := LoadPositions(filepath.Join(dir, "positions.json"))
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	go TailFiles(ctx, src, pos2, &Tracker{}, out)
	if got := collect(t, out, 1); got[0] != "while down" {
		t.Fatalf("resume: %q", got)
	}
}

func TestDockerFrames(t *testing.T) {
	var buf bytes.Buffer
	frame := func(stream byte, s string) {
		hdr := make([]byte, 8)
		hdr[0] = stream
		binary.BigEndian.PutUint32(hdr[4:], uint32(len(s)))
		buf.Write(hdr)
		buf.WriteString(s)
	}
	frame(1, "2024-01-01T00:00:00.000000000Z hello\n2024-01-01T00:00:01Z wor")
	frame(2, "2024-01-01T00:00:02Z err\n")
	frame(1, "ld\n")
	var got []string
	readDockerStream(&buf, false, func(stream, text string) bool {
		_, msg := splitDockerTime(text)
		got = append(got, stream+":"+msg)
		return true
	})
	want := []string{"stdout:hello", "stderr:err", "stdout:world"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q", got)
	}
}

func TestParseSyslog(t *testing.T) {
	msg, app, tags := parseSyslog("<34>Oct 11 22:14:15 gw su[12]: 'su root' failed")
	if app != "su" || msg != "Oct 11 22:14:15 gw su[12]: 'su root' failed" || tags[0] != "severity:crit" {
		t.Fatalf("3164: %q %q %q", msg, app, tags)
	}
	msg, app, _ = parseSyslog("<165>1 2003-10-11T22:14:15.003Z host evntslog - ID47 [exampleSDID@32473 iut=\"3\"] An application event")
	if app != "evntslog" || msg != "An application event" {
		t.Fatalf("5424: %q %q", msg, app)
	}
}

func TestParseJournal(t *testing.T) {
	l, cursor, ok := parseJournal([]byte(`{"MESSAGE":"hi","_SYSTEMD_UNIT":"nginx.service","SYSLOG_IDENTIFIER":"nginx","__CURSOR":"c1","__REALTIME_TIMESTAMP":"1700000000000000"}`))
	if !ok || l.Text != "hi" || l.Service != "nginx" || cursor != "c1" || l.Time.Unix() != 1700000000 {
		t.Fatalf("got %+v %q", l, cursor)
	}
	if _, _, ok := parseJournal([]byte(`{"MESSAGE":[1,2]}`)); ok {
		t.Fatal("binary message should be skipped")
	}
}
