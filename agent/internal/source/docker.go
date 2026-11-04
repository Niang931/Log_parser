package source

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/jakekato/logpipe/agent/internal/config"
)

// DockerSocket is the Engine API socket; mount it read-only into the agent container.
func DockerSocket() string {
	if h := os.Getenv("DOCKER_HOST"); strings.HasPrefix(h, "unix://") {
		return strings.TrimPrefix(h, "unix://")
	}
	return "/var/run/docker.sock"
}

func dockerClient() *http.Client {
	sock := DockerSocket()
	return &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", sock)
		},
	}}
}

type container struct {
	ID     string            `json:"Id"`
	Names  []string          `json:"Names"`
	Image  string            `json:"Image"`
	Labels map[string]string `json:"Labels"`
}

// Docker follows stdout/stderr of every running container (optionally only those with all of
// container_labels, written key=value), picking up new containers every 5s.
func Docker(ctx context.Context, src config.Source, pos *Positions, tr *Tracker, out chan<- Line) error {
	client := dockerClient()
	var mu sync.Mutex
	following := map[string]bool{}
	self, _ := os.Hostname() // inside a container this is the short container id: skip our own logs
	for ctx.Err() == nil {
		list, err := listContainers(ctx, client, src.Labels)
		tr.SetError(err)
		var names []string
		for _, c := range list {
			name := strings.TrimPrefix(firstOr(c.Names, c.ID[:12]), "/")
			names = append(names, name)
			mu.Lock()
			skip := following[c.ID] || strings.HasPrefix(c.ID, self)
			following[c.ID] = true
			mu.Unlock()
			if skip {
				continue
			}
			go func(c container, name string) {
				err := followContainer(ctx, client, c, name, src, pos, tr, out)
				if err != nil && ctx.Err() == nil {
					tr.SetError(fmt.Errorf("%s: %w", name, err))
				}
				mu.Lock()
				delete(following, c.ID) // re-attach on the next scan if it is still running
				mu.Unlock()
			}(c, name)
		}
		tr.SetFiles(names)
		select {
		case <-ctx.Done():
		case <-time.After(5 * time.Second):
		}
	}
	return nil
}

func firstOr(xs []string, d string) string {
	if len(xs) > 0 {
		return xs[0]
	}
	return d
}

func listContainers(ctx context.Context, client *http.Client, labels []string) ([]container, error) {
	q := url.Values{}
	if len(labels) > 0 {
		f, _ := json.Marshal(map[string][]string{"label": labels})
		q.Set("filters", string(f))
	}
	req, _ := http.NewRequestWithContext(ctx, "GET", "http://docker/containers/json?"+q.Encode(), nil)
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("docker: %s", resp.Status)
	}
	var list []container
	return list, json.NewDecoder(resp.Body).Decode(&list)
}

func followContainer(ctx context.Context, client *http.Client, c container, name string, src config.Source, pos *Positions, tr *Tracker, out chan<- Line) error {
	key := "docker:" + c.ID
	since := "0"
	if p, ok := pos.Get(key); ok && p.Cursor != "" {
		since = p.Cursor
	} else if src.StartAt != "beginning" {
		since = fmt.Sprint(time.Now().Unix())
	}
	req, _ := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("http://docker/containers/%s/logs?follow=1&stdout=1&stderr=1&timestamps=1&since=%s", c.ID, since), nil)
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("logs: %s", resp.Status)
	}
	tags := []string{"container_name:" + name, "image:" + c.Image}
	return readDockerStream(resp.Body, resp.Header.Get("Content-Type") == "application/vnd.docker.raw-stream", func(stream, text string) bool {
		ts, msg := splitDockerTime(text)
		tr.Count(msg)
		ok := send(ctx, out, Line{Key: key + ":" + stream, Text: msg, Time: ts, Service: name, Tags: append(tags, "stream:"+stream)})
		// Docker's since is in seconds, so a restart may resend lines from the last second.
		pos.Set(key, Position{Cursor: fmt.Sprint(ts.Unix())})
		return ok
	})
}

// readDockerStream splits the Engine log stream: TTY containers send raw text, the rest use
// 8-byte frame headers [stream, 0, 0, 0, size(4, big endian)].
func readDockerStream(r io.Reader, raw bool, emit func(stream, text string) bool) error {
	if raw {
		sc := bufio.NewScanner(r)
		sc.Buffer(make([]byte, 64*1024), maxLine)
		for sc.Scan() {
			if !emit("stdout", sc.Text()) {
				return nil
			}
		}
		return sc.Err()
	}
	br := bufio.NewReader(r)
	hdr := make([]byte, 8)
	partial := map[byte]string{}
	for {
		if _, err := io.ReadFull(br, hdr); err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		size := binary.BigEndian.Uint32(hdr[4:])
		buf := make([]byte, size)
		if _, err := io.ReadFull(br, buf); err != nil {
			return err
		}
		stream := "stdout"
		if hdr[0] == 2 {
			stream = "stderr"
		}
		text := partial[hdr[0]] + string(buf)
		lines := strings.Split(text, "\n")
		partial[hdr[0]] = lines[len(lines)-1]
		for _, l := range lines[:len(lines)-1] {
			if !emit(stream, l) {
				return nil
			}
		}
	}
}

func splitDockerTime(s string) (time.Time, string) {
	if i := strings.IndexByte(s, ' '); i > 0 {
		if ts, err := time.Parse(time.RFC3339Nano, s[:i]); err == nil {
			return ts, s[i+1:]
		}
	}
	return time.Now(), s
}
