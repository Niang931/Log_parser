package source

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"time"

	"github.com/jakekato/logpipe/agent/internal/config"
)

// Journald follows `journalctl -o json -f`, resuming after the saved cursor.
// shortcut: shells out to journalctl instead of linking libsystemd (no cgo, static binary); needs journalctl on the host or in the image.
func Journald(ctx context.Context, src config.Source, pos *Positions, tr *Tracker, out chan<- Line) error {
	key := "journald:" + src.Unit
	for ctx.Err() == nil {
		args := []string{"-o", "json", "-f", "--no-pager"}
		if p, ok := pos.Get(key); ok && p.Cursor != "" {
			args = append(args, "--after-cursor", p.Cursor)
		} else if src.StartAt == "beginning" {
			args = append(args, "--no-tail")
		} else {
			args = append(args, "-n", "0")
		}
		if src.Unit != "" {
			args = append(args, "-u", src.Unit)
		}
		err := runJournal(ctx, args, key, pos, tr, out)
		tr.SetError(err)
		select {
		case <-ctx.Done():
		case <-time.After(5 * time.Second): // journalctl exited; retry
		}
	}
	return nil
}

func runJournal(ctx context.Context, args []string, key string, pos *Positions, tr *Tracker, out chan<- Line) error {
	cmd := exec.CommandContext(ctx, "journalctl", args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("journalctl: %w", err)
	}
	sc := bufio.NewScanner(stdout)
	sc.Buffer(make([]byte, 64*1024), maxLine)
	for sc.Scan() {
		l, cursor, ok := parseJournal(sc.Bytes())
		if !ok {
			continue
		}
		tr.Count(l.Text)
		if !send(ctx, out, l) {
			break
		}
		pos.Set(key, Position{Cursor: cursor})
	}
	return cmd.Wait()
}

func parseJournal(b []byte) (Line, string, bool) {
	var e map[string]any
	if json.Unmarshal(b, &e) != nil {
		return Line{}, "", false
	}
	msg, ok := e["MESSAGE"].(string)
	if !ok {
		return Line{}, "", false // binary MESSAGE arrives as a byte array; skip it
	}
	unit, _ := e["_SYSTEMD_UNIT"].(string)
	ident, _ := e["SYSLOG_IDENTIFIER"].(string)
	cursor, _ := e["__CURSOR"].(string)
	ts := time.Now()
	if s, ok := e["__REALTIME_TIMESTAMP"].(string); ok {
		if us, err := strconv.ParseInt(s, 10, 64); err == nil {
			ts = time.UnixMicro(us)
		}
	}
	service := ident
	if service == "" {
		service = unit
	}
	key := unit
	if key == "" {
		key = ident
	}
	return Line{Key: "journald:" + key, Text: msg, Time: ts, Service: service, Tags: []string{"unit:" + unit}}, cursor, true
}
