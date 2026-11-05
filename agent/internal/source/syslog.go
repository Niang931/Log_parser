package source

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"regexp"
	"strings"
	"time"

	"github.com/jakekato/logpipe/agent/internal/config"
)

var priority = regexp.MustCompile(`^<(\d{1,3})>`)

// rfc5424 header after the priority: VERSION TIMESTAMP HOST APP PROCID MSGID
var rfc5424 = regexp.MustCompile(`^1 \S+ \S+ (\S+) \S+ \S+ (?:-|\[.*?\]) ?`)

// rfc3164 header: "Jan  2 03:04:05 host app[pid]: "
var rfc3164 = regexp.MustCompile(`^[A-Z][a-z]{2} [ \d]\d \d{2}:\d{2}:\d{2} \S+ ([^:\[\s]+)(?:\[\d+\])?: ?`)

var severities = []string{"emerg", "alert", "crit", "err", "warning", "notice", "info", "debug"}

// parseSyslog keeps the message text and turns the header into app + severity, leaving the timestamp
// in front so the server's timestamp extraction still sees it for RFC 3164.
func parseSyslog(s string) (msg, app string, tags []string) {
	s = strings.TrimRight(s, "\r\n\x00")
	if m := priority.FindStringSubmatch(s); m != nil {
		var pri int
		fmt.Sscan(m[1], &pri)
		tags = append(tags, "severity:"+severities[pri%8])
		s = s[len(m[0]):]
	}
	if m := rfc5424.FindStringSubmatch(s); m != nil {
		return s[len(m[0]):], strings.Trim(m[1], "-"), tags
	}
	if m := rfc3164.FindStringSubmatch(s); m != nil {
		return s, m[1], tags
	}
	return s, "", tags
}

// Syslog listens on TCP (newline-framed) or UDP (one message per datagram).
// shortcut: TCP octet-counting framing (RFC 6587) is not parsed; most senders default to newlines.
func Syslog(ctx context.Context, src config.Source, _ *Positions, tr *Tracker, out chan<- Line) error {
	addr := fmt.Sprintf(":%d", src.Port)
	emit := func(peer, raw string) bool {
		msg, app, tags := parseSyslog(raw)
		if strings.TrimSpace(msg) == "" {
			return true
		}
		tr.Count(msg)
		return send(ctx, out, Line{Key: "syslog:" + peer + ":" + app, Text: msg, Time: time.Now(), Service: app,
			Tags: append(tags, "peer:"+peer)})
	}
	if src.Type == "udp" {
		pc, err := net.ListenPacket("udp", addr)
		if err != nil {
			tr.SetError(err)
			return err
		}
		go func() { <-ctx.Done(); pc.Close() }()
		buf := make([]byte, 64*1024)
		for {
			n, from, err := pc.ReadFrom(buf)
			if err != nil {
				return nil
			}
			host, _, _ := net.SplitHostPort(from.String())
			for _, l := range strings.Split(string(buf[:n]), "\n") {
				if !emit(host, l) {
					return nil
				}
			}
		}
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		tr.SetError(err)
		return err
	}
	go func() { <-ctx.Done(); ln.Close() }()
	for {
		conn, err := ln.Accept()
		if err != nil {
			return nil
		}
		go func() {
			defer conn.Close()
			go func() { <-ctx.Done(); conn.Close() }()
			host, _, _ := net.SplitHostPort(conn.RemoteAddr().String())
			sc := bufio.NewScanner(conn)
			sc.Buffer(make([]byte, 64*1024), maxLine)
			for sc.Scan() {
				if !emit(host, sc.Text()) {
					return
				}
			}
		}()
	}
}
