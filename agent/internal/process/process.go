// Package process turns raw lines from one source into events: multiline aggregation first,
// then exclude/include/mask rules in the order they are written.
package process

import (
	"regexp"
	"strings"
	"time"

	"github.com/jakekato/logpipe/agent/internal/config"
)

// MaxEvent caps one aggregated event (a runaway stack trace) at 256KB.
const MaxEvent = 256 * 1024

// FlushAfter is how long a multiline event may wait for more lines before it is sent.
const FlushAfter = time.Second

// autoPatterns are the line starts auto_multi_line_detection knows; the first one a line
// matches becomes the source's start pattern.
var autoPatterns = []*regexp.Regexp{
	regexp.MustCompile(`^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}`),                    // 2024-01-02 03:04:05 / ISO8601
	regexp.MustCompile(`^\[?\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}`),                    // 2024/01/02 03:04:05
	regexp.MustCompile(`^\[?\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}`),            // 02/Jan/2024:03:04:05 (nginx)
	regexp.MustCompile(`^[A-Z][a-z]{2} [ \d]\d \d{2}:\d{2}:\d{2}`),                   // Jan  2 03:04:05 (syslog)
	regexp.MustCompile(`^\[?\d{2}:\d{2}:\d{2}[.,]\d+`),                               // 03:04:05.123
	regexp.MustCompile(`^\d{10}(\.\d+)?\s`),                                          // epoch seconds
	regexp.MustCompile(`^\[?(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b`), // level first
	regexp.MustCompile(`^\{`),                                                        // JSON per line
}

type filter struct {
	kind    string
	re      *regexp.Regexp
	replace string
}

type Processor struct {
	start   *regexp.Regexp
	auto    bool
	filters []filter
	pending strings.Builder
	meta    any
	has     bool
	last    time.Time
}

// Event is a finished event with the metadata of its first line.
type Event struct {
	Text string
	Meta any
}

func New(src config.Source, autoDefault bool) *Processor {
	p := &Processor{auto: autoDefault}
	if src.AutoMultiLine != nil {
		p.auto = *src.AutoMultiLine
	}
	for _, r := range src.Rules {
		re := regexp.MustCompile(r.Pattern) // validated on load
		if r.Type == "multi_line" {
			p.start, p.auto = re, false
			continue
		}
		p.filters = append(p.filters, filter{r.Type, re, r.Replacement})
	}
	return p
}

// Line feeds one raw line (with caller metadata such as its time) and returns the events it completed.
func (p *Processor) Line(line string, meta any, now time.Time) []Event {
	line = strings.TrimRight(line, "\r\n")
	if p.start == nil && p.auto {
		for _, re := range autoPatterns {
			if re.MatchString(line) {
				p.start = re
				break
			}
		}
	}
	if p.start == nil {
		// One line, one event; drop blank lines.
		if strings.TrimSpace(line) == "" {
			return nil
		}
		return p.apply(line, meta)
	}
	var out []Event
	if p.start.MatchString(line) {
		out = p.Flush()
	} else if !p.has {
		// A continuation with nothing to continue (start of a file mid-trace): send it alone.
		if strings.TrimSpace(line) == "" {
			return nil
		}
		return p.apply(line, meta)
	}
	if p.has {
		if p.pending.Len() < MaxEvent {
			p.pending.WriteByte('\n')
			p.pending.WriteString(line)
		}
	} else {
		p.pending.WriteString(line)
		p.meta, p.has = meta, true
	}
	p.last = now
	return out
}

// Due reports whether a pending multiline event has waited long enough to be sent.
func (p *Processor) Due(now time.Time) bool { return p.has && now.Sub(p.last) >= FlushAfter }

func (p *Processor) Flush() []Event {
	if !p.has {
		return nil
	}
	ev := p.pending.String()
	if len(ev) > MaxEvent {
		ev = ev[:MaxEvent]
	}
	p.pending.Reset()
	p.has = false
	return p.apply(ev, p.meta)
}

func (p *Processor) apply(ev string, meta any) []Event {
	for _, f := range p.filters {
		switch f.kind {
		case "exclude_at_match":
			if f.re.MatchString(ev) {
				return nil
			}
		case "include_at_match":
			if !f.re.MatchString(ev) {
				return nil
			}
		case "mask_sequences":
			ev = f.re.ReplaceAllString(ev, f.replace)
		}
	}
	return []Event{{ev, meta}}
}
