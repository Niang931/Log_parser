package source

import (
	"bufio"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/jakekato/logpipe/agent/internal/config"
)

// maxLine splits a line with no newline in sight (a binary file) so memory stays bounded.
const maxLine = 1 << 20

// PollEvery is how often globs are re-expanded and files checked for growth or rotation.
var PollEvery = 250 * time.Millisecond

// shortcut: polling instead of inotify costs a stat per file per 250ms; fine for tens of files, switch to fsnotify for thousands.

type tailed struct {
	f       *os.File
	r       *bufio.Reader
	inode   uint64
	offset  int64
	partial []byte
}

func inodeOf(fi os.FileInfo) uint64 {
	if st, ok := fi.Sys().(*syscall.Stat_t); ok {
		return st.Ino
	}
	return 0
}

// TailFiles follows every file matching src.Path, surviving rotation (rename + new file) and
// truncation (copytruncate). Files present at the first scan start at their saved offset, or at the
// end unless start_position is "beginning"; files that appear later are read from the start.
func TailFiles(ctx context.Context, src config.Source, pos *Positions, tr *Tracker, out chan<- Line) error {
	open := map[string]*tailed{}
	defer func() {
		for _, t := range open {
			t.f.Close()
		}
	}()
	first := true
	tick := time.NewTicker(PollEvery)
	defer tick.Stop()
	for {
		matches, err := filepath.Glob(src.Path)
		tr.SetError(err)
		sort.Strings(matches)
		tr.SetFiles(matches)
		seen := map[string]bool{}
		for _, path := range matches {
			seen[path] = true
			fi, err := os.Stat(path)
			if err != nil || fi.IsDir() {
				continue
			}
			t := open[path]
			if t != nil && t.inode != inodeOf(fi) {
				// Rotated: drain the old file, then follow the new one from its start.
				if !drain(ctx, path, t, tr, out) {
					return nil
				}
				t.f.Close()
				t = nil
				delete(open, path)
				if t, err = openAt(path, 0, fi); err == nil {
					open[path] = t
				}
			}
			if t == nil {
				off := int64(0)
				if p, ok := pos.Get(path); ok && p.Inode == inodeOf(fi) && p.Offset <= fi.Size() {
					off = p.Offset
				} else if first && src.StartAt != "beginning" {
					off = fi.Size()
				}
				if t, err = openAt(path, off, fi); err != nil {
					tr.SetError(err)
					continue
				}
				open[path] = t
			}
			if fi.Size() < t.offset { // truncated in place
				t.f.Seek(0, io.SeekStart)
				t.r.Reset(t.f)
				t.offset, t.partial = 0, nil
			}
			if !drain(ctx, path, t, tr, out) {
				return nil
			}
			pos.Set(path, Position{Inode: t.inode, Offset: t.offset})
		}
		for path, t := range open {
			if !seen[path] { // deleted or no longer matching
				drain(ctx, path, t, tr, out)
				t.f.Close()
				delete(open, path)
			}
		}
		first = false
		select {
		case <-ctx.Done():
			return nil
		case <-tick.C:
		}
	}
}

func openAt(path string, off int64, fi os.FileInfo) (*tailed, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	if _, err := f.Seek(off, io.SeekStart); err != nil {
		f.Close()
		return nil, err
	}
	return &tailed{f: f, r: bufio.NewReaderSize(f, 64*1024), inode: inodeOf(fi), offset: off}, nil
}

// drain sends every complete line; a trailing partial line waits for its newline. The offset only
// moves past complete lines, so a restart re-reads a half-written line instead of splitting it.
func drain(ctx context.Context, path string, t *tailed, tr *Tracker, out chan<- Line) bool {
	for {
		chunk, err := t.r.ReadSlice('\n')
		if len(chunk) > 0 {
			t.partial = append(t.partial, chunk...)
		}
		if errors.Is(err, bufio.ErrBufferFull) && len(t.partial) < maxLine {
			continue
		}
		if err != nil && !errors.Is(err, bufio.ErrBufferFull) {
			return true // EOF: keep partial for the next poll
		}
		line := strings.TrimSuffix(string(t.partial), "\n")
		t.offset += int64(len(t.partial))
		t.partial = t.partial[:0]
		tr.Count(line)
		if !send(ctx, out, Line{Key: path, Text: line, Time: time.Now(), Tags: []string{"filename:" + filepath.Base(path)}}) {
			return false
		}
	}
}
