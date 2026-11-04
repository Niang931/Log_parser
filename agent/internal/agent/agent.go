// Package agent wires config, sources, processing and shipping together, reports heartbeats and
// pulls server-pushed config into conf.d/remote.yaml. Any config change on disk reloads the sources.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/jakekato/logpipe/agent/internal/config"
	"github.com/jakekato/logpipe/agent/internal/process"
	"github.com/jakekato/logpipe/agent/internal/ship"
	"github.com/jakekato/logpipe/agent/internal/source"
)

var (
	HeartbeatEvery = 15 * time.Second
	WatchEvery     = 2 * time.Second
)

// Status is written to run/status.json for `logpipe-agent status` and sent as the heartbeat.
type Status struct {
	Hostname    string          `json:"hostname"`
	Version     string          `json:"version"`
	OS          string          `json:"os"`
	Arch        string          `json:"arch"`
	StartedAt   string          `json:"started_at"`
	Endpoint    string          `json:"endpoint"`
	Tags        []string        `json:"tags"`
	Sources     []source.Stats  `json:"sources"`
	Shipper     ship.Stats      `json:"shipper"`
	Config      []config.Source `json:"config"`
	RemoteHash  string          `json:"remote_hash"`
	ConfigError string          `json:"config_error,omitempty"`
	Heartbeat   string          `json:"last_heartbeat_at,omitempty"`
	Updated     string          `json:"updated_at"`
}

func StatusPath(cfg *config.Config) string { return filepath.Join(cfg.RunPath, "status.json") }

// Run keeps the agent going until ctx is done, reloading whenever the config files change.
func Run(ctx context.Context, dir, version string) error {
	started := time.Now()
	// Counters survive reloads, so a pushed config doesn't zero the UI's numbers.
	trackers := map[string]*source.Tracker{}
	var shipped ship.Stats
	for ctx.Err() == nil {
		cfg, err := config.Load(dir)
		if err != nil {
			return err
		}
		log.Printf("running %d sources, shipping to %s as %s", len(cfg.Sources), cfg.Endpoint, cfg.Hostname)
		reload, err := runOnce(ctx, cfg, version, started, trackers, &shipped)
		if err != nil {
			return err
		}
		if reload {
			log.Printf("config changed, reloading")
		}
	}
	return nil
}

func runOnce(parent context.Context, cfg *config.Config, version string, started time.Time,
	known map[string]*source.Tracker, shipped *ship.Stats) (bool, error) {
	if err := os.MkdirAll(cfg.RunPath, 0o755); err != nil {
		return false, err
	}
	pos := source.LoadPositions(filepath.Join(cfg.RunPath, "positions.json"))
	client := ship.NewClient(cfg.Endpoint, cfg.APIKey, version)
	shipper := ship.New(client, filepath.Join(cfg.RunPath, "buffer"), cfg.BufferMB)
	shipper.Base = *shipped

	srcCtx, stopSources := context.WithCancel(parent)
	defer stopSources()
	shipCtx, stopShip := context.WithCancel(context.WithoutCancel(parent))
	defer stopShip()

	events := make(chan ship.Event, 4*ship.BatchSize)
	shipDone := make(chan struct{})
	go func() { shipper.Run(shipCtx, events); close(shipDone) }()

	trackers := make([]*source.Tracker, len(cfg.Sources))
	var wg sync.WaitGroup
	for i, src := range cfg.Sources {
		if known[src.ID()] == nil {
			known[src.ID()] = &source.Tracker{}
		}
		trackers[i] = known[src.ID()]
		lines := make(chan source.Line, 1024)
		wg.Add(2)
		go func(src config.Source, tr *source.Tracker) {
			defer wg.Done()
			defer close(lines)
			if err := source.For(src)(srcCtx, src, pos, tr, lines); err != nil {
				log.Printf("source %s: %v", src.ID(), err)
			}
		}(src, trackers[i])
		go func(src config.Source) {
			defer wg.Done()
			processLines(cfg, src, lines, events)
		}(src)
	}

	status := func(hb string, cfgErr string) Status {
		st := Status{Hostname: cfg.Hostname, Version: version, OS: runtime.GOOS, Arch: runtime.GOARCH,
			StartedAt: started.UTC().Format(time.RFC3339), Endpoint: cfg.Endpoint, Tags: cfg.Tags,
			Shipper: shipper.Stats(), Config: cfg.Sources, RemoteHash: remoteHash(cfg.Dir),
			ConfigError: cfgErr, Heartbeat: hb, Updated: time.Now().UTC().Format(time.RFC3339)}
		for i, src := range cfg.Sources {
			st.Sources = append(st.Sources, trackers[i].Snapshot(src))
		}
		return st
	}

	fingerprint := configFingerprint(cfg.Dir)
	watch := time.NewTicker(WatchEvery)
	defer watch.Stop()
	beat := time.NewTicker(HeartbeatEvery)
	defer beat.Stop()
	var lastBeat, cfgErr string
	heartbeat := func() {
		if err := sendHeartbeat(parent, client, cfg, status(lastBeat, cfgErr)); err != nil {
			log.Printf("heartbeat: %v", err)
		} else {
			lastBeat = time.Now().UTC().Format(time.RFC3339)
		}
	}
	heartbeat()
	reload := false
loop:
	for {
		select {
		case <-parent.Done():
			break loop
		case <-beat.C:
			heartbeat()
		case <-watch.C:
			pos.Save()
			writeStatus(cfg, status(lastBeat, cfgErr))
			if fp := configFingerprint(cfg.Dir); fp != fingerprint {
				if _, err := config.Load(cfg.Dir); err != nil {
					cfgErr = err.Error() // keep running the last good config
					log.Printf("ignoring invalid config change: %v", err)
				} else {
					reload = true
					break loop
				}
				fingerprint = fp
			}
		}
	}
	// Stop reading, let processors flush their pending multiline events, then drain the shipper.
	stopSources()
	wg.Wait()
	close(events)
	select {
	case <-shipDone:
	case <-time.After(35 * time.Second):
	}
	pos.Save()
	writeStatus(cfg, status(lastBeat, cfgErr))
	st := shipper.Stats()
	shipped.Sent, shipped.Dropped, shipped.LastSend = st.Sent, st.Dropped, st.LastSend
	return reload, nil
}

// processLines runs one source's lines through its rules; each Key (file, container, peer) gets
// its own multiline state.
func processLines(cfg *config.Config, src config.Source, in <-chan source.Line, out chan<- ship.Event) {
	procs := map[string]*process.Processor{}
	emit := func(evs []process.Event) {
		for _, e := range evs {
			out <- event(cfg, src, e.Meta.(source.Line), e.Text)
		}
	}
	tick := time.NewTicker(process.FlushAfter / 4)
	defer tick.Stop()
	for {
		select {
		case l, ok := <-in:
			if !ok {
				for _, p := range procs {
					emit(p.Flush())
				}
				return
			}
			p := procs[l.Key]
			if p == nil {
				p = process.New(src, cfg.AutoMultiLine)
				procs[l.Key] = p
			}
			emit(p.Line(l.Text, l, time.Now()))
		case now := <-tick.C:
			for _, p := range procs {
				if p.Due(now) {
					emit(p.Flush())
				}
			}
		}
	}
}

func event(cfg *config.Config, src config.Source, l source.Line, msg string) ship.Event {
	service := src.Service
	if service == "" {
		service = l.Service
	}
	if service == "" && src.Type == "file" {
		service = strings.TrimSuffix(filepath.Base(l.Key), filepath.Ext(l.Key))
	}
	if service == "" {
		service = src.Type
	}
	name := src.Src
	if name == "" {
		name = src.Type
	}
	tags := append(append(append([]string{}, cfg.Tags...), src.Tags...), l.Tags...)
	// Local time with its offset: the server reads zone-less timestamps in the line in this zone.
	return ship.Event{Time: l.Time.Format(time.RFC3339Nano), Host: cfg.Hostname, Service: service,
		Source: name, Tags: tags, Message: msg}
}

func writeStatus(cfg *config.Config, st Status) {
	b, _ := json.MarshalIndent(st, "", "  ")
	tmp := StatusPath(cfg) + ".tmp"
	if os.WriteFile(tmp, b, 0o644) == nil {
		os.Rename(tmp, StatusPath(cfg))
	}
}

func configFingerprint(dir string) string {
	files, _ := filepath.Glob(filepath.Join(dir, "conf.d", "*.yaml"))
	files = append(files, filepath.Join(dir, "agent.yaml"))
	sort.Strings(files)
	var all []byte
	for _, f := range files {
		b, _ := os.ReadFile(f)
		all = append(append(all, f...), b...)
	}
	return config.HashBytes(all)
}

func remoteHash(dir string) string {
	b, err := os.ReadFile(filepath.Join(dir, "conf.d", config.RemoteFile))
	if err != nil {
		return ""
	}
	return config.HashBytes(b)
}

// sendHeartbeat reports status; if the server holds a different remote config, it is fetched,
// validated and written to conf.d/remote.yaml, which the watcher then reloads.
func sendHeartbeat(ctx context.Context, client *ship.Client, cfg *config.Config, st Status) error {
	body, _ := json.Marshal(st)
	hbCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	resp, err := client.Do(hbCtx, "POST", "/intake/v1/heartbeat", body, false)
	if err != nil {
		return err
	}
	var ans struct {
		RemoteHash *string `json:"remote_hash"`
	}
	json.Unmarshal(resp, &ans)
	if !cfg.RemoteEnabled() || ans.RemoteHash == nil || *ans.RemoteHash == st.RemoteHash {
		return nil
	}
	raw, err := client.Do(hbCtx, "GET", "/intake/v1/config?hostname="+url.QueryEscape(cfg.Hostname), nil, false)
	if err != nil {
		return fmt.Errorf("fetch remote config: %w", err)
	}
	var rc struct {
		YAML string `json:"yaml"`
	}
	if err := json.Unmarshal(raw, &rc); err != nil {
		return err
	}
	path := filepath.Join(cfg.Dir, "conf.d", config.RemoteFile)
	if strings.TrimSpace(rc.YAML) == "" {
		os.Remove(path)
		return nil
	}
	if err := config.CheckFile(path, []byte(rc.YAML)); err != nil {
		return fmt.Errorf("rejected remote config: %w", err)
	}
	os.MkdirAll(filepath.Dir(path), 0o755)
	log.Printf("applying remote config %s", config.HashBytes([]byte(rc.YAML)))
	return os.WriteFile(path, []byte(rc.YAML), 0o644)
}
