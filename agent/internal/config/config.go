// Package config loads the Datadog-style agent configuration: a main agent.yaml plus
// conf.d/*.yaml files that each hold a `logs:` list of sources.
package config

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// RemoteFile is the conf.d file the agent owns; it is overwritten by configs pushed from the server.
const RemoteFile = "remote.yaml"

type Agent struct {
	APIKey   string   `yaml:"api_key" json:"-"`
	Endpoint string   `yaml:"endpoint" json:"endpoint"`
	Hostname string   `yaml:"hostname,omitempty" json:"hostname"`
	Tags     []string `yaml:"tags,omitempty" json:"tags"`
	// RunPath holds offsets and the on-disk send buffer.
	RunPath string `yaml:"run_path,omitempty" json:"run_path"`
	// BufferMB caps the on-disk buffer used while the server is unreachable.
	BufferMB int `yaml:"buffer_mb,omitempty" json:"buffer_mb"`
	// RemoteConfig lets the server push sources into conf.d/remote.yaml.
	RemoteConfig *bool `yaml:"remote_config,omitempty" json:"remote_config"`
	// AutoMultiLine turns on automatic multiline detection for every source without an explicit rule.
	AutoMultiLine bool `yaml:"auto_multi_line_detection,omitempty" json:"auto_multi_line_detection"`
}

type Rule struct {
	Type        string `yaml:"type" json:"type"`
	Name        string `yaml:"name,omitempty" json:"name,omitempty"`
	Pattern     string `yaml:"pattern" json:"pattern"`
	Replacement string `yaml:"replace_placeholder,omitempty" json:"replace_placeholder,omitempty"`
}

type Source struct {
	Type    string   `yaml:"type" json:"type"`
	Path    string   `yaml:"path,omitempty" json:"path,omitempty"`
	Port    int      `yaml:"port,omitempty" json:"port,omitempty"`
	Proto   string   `yaml:"protocol,omitempty" json:"protocol,omitempty"`
	Unit    string   `yaml:"unit,omitempty" json:"unit,omitempty"`
	Labels  []string `yaml:"container_labels,omitempty" json:"container_labels,omitempty"`
	Service string   `yaml:"service,omitempty" json:"service,omitempty"`
	Src     string   `yaml:"source,omitempty" json:"source,omitempty"`
	Tags    []string `yaml:"tags,omitempty" json:"tags,omitempty"`
	// StartAt is "end" (default, like tail -f) or "beginning" for files seen for the first time.
	StartAt       string `yaml:"start_position,omitempty" json:"start_position,omitempty"`
	AutoMultiLine *bool  `yaml:"auto_multi_line_detection,omitempty" json:"auto_multi_line_detection,omitempty"`
	Rules         []Rule `yaml:"log_processing_rules,omitempty" json:"log_processing_rules,omitempty"`
	// File is the conf.d file this source came from (filled on load).
	File string `yaml:"-" json:"file"`
}

type Integration struct {
	Logs []Source `yaml:"logs"`
}

type Config struct {
	Agent
	Dir     string   `json:"-"`
	Sources []Source `json:"sources"`
}

var SourceTypes = map[string]bool{"file": true, "journald": true, "docker": true, "tcp": true, "udp": true}
var RuleTypes = map[string]bool{"multi_line": true, "exclude_at_match": true, "include_at_match": true, "mask_sequences": true}

var envRef = regexp.MustCompile(`\$\{([A-Za-z_][A-Za-z0-9_]*)\}`)

func expand(b []byte) []byte {
	return envRef.ReplaceAllFunc(b, func(m []byte) []byte {
		return []byte(os.Getenv(string(envRef.FindSubmatch(m)[1])))
	})
}

// DefaultDir is where the agent looks for agent.yaml unless -c or LOGPIPE_AGENT_CONFIG says
// otherwise: /etc/logpipe-agent for root, the user's config directory for everyone else, so
// the agent never needs sudo just to find its config.
func DefaultDir() string {
	if d := os.Getenv("LOGPIPE_AGENT_CONFIG"); d != "" {
		return d
	}
	return defaultDir(os.Geteuid(), os.UserConfigDir)
}

func defaultDir(euid int, userConfig func() (string, error)) string {
	if euid == 0 {
		return "/etc/logpipe-agent"
	}
	if d, err := userConfig(); err == nil {
		return filepath.Join(d, "logpipe-agent")
	}
	return "/etc/logpipe-agent"
}

func Load(dir string) (*Config, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "agent.yaml"))
	if err != nil {
		return nil, err
	}
	cfg := &Config{Dir: dir}
	if err := yaml.Unmarshal(expand(raw), &cfg.Agent); err != nil {
		return nil, fmt.Errorf("agent.yaml: %w", err)
	}
	// Environment wins over the file so the docker image works with `-e` alone.
	if v := os.Getenv("LOGPIPE_API_KEY"); v != "" {
		cfg.APIKey = v
	}
	if v := os.Getenv("LOGPIPE_ENDPOINT"); v != "" {
		cfg.Endpoint = v
	}
	if v := os.Getenv("LOGPIPE_HOSTNAME"); v != "" {
		cfg.Hostname = v
	}
	if cfg.Hostname == "" {
		cfg.Hostname, _ = os.Hostname()
	}
	if cfg.RunPath == "" {
		cfg.RunPath = filepath.Join(dir, "run")
	}
	if cfg.BufferMB == 0 {
		cfg.BufferMB = 100
	}
	files, _ := filepath.Glob(filepath.Join(dir, "conf.d", "*.yaml"))
	sort.Strings(files)
	for _, f := range files {
		integ, err := ReadIntegration(f)
		if err != nil {
			return nil, err
		}
		for _, s := range integ.Logs {
			s.File = filepath.Base(f)
			cfg.Sources = append(cfg.Sources, s)
		}
	}
	return cfg, cfg.Validate()
}

func ReadIntegration(path string) (*Integration, error) {
	integ := &Integration{}
	raw, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return integ, nil
	}
	if err != nil {
		return nil, err
	}
	if err := yaml.Unmarshal(expand(raw), integ); err != nil {
		return nil, fmt.Errorf("%s: %w", filepath.Base(path), err)
	}
	return integ, nil
}

func WriteIntegration(path string, integ *Integration) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if len(integ.Logs) == 0 {
		err := os.Remove(path)
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	out, err := yaml.Marshal(integ)
	if err != nil {
		return err
	}
	return writeAtomic(path, out)
}

func writeAtomic(path string, data []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (c *Config) Validate() error {
	var errs []string
	if c.Endpoint == "" {
		errs = append(errs, "endpoint is required")
	}
	for i, s := range c.Sources {
		if err := s.Validate(); err != nil {
			errs = append(errs, fmt.Sprintf("%s logs[%d]: %v", s.File, i, err))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("invalid config:\n  %s", strings.Join(errs, "\n  "))
	}
	return nil
}

func (s Source) Validate() error {
	if !SourceTypes[s.Type] {
		return fmt.Errorf("unknown type %q (file, journald, docker, tcp, udp)", s.Type)
	}
	switch s.Type {
	case "file":
		if s.Path == "" {
			return fmt.Errorf("file source needs path")
		}
	case "tcp", "udp":
		if s.Port <= 0 || s.Port > 65535 {
			return fmt.Errorf("%s source needs a port", s.Type)
		}
	}
	if s.StartAt != "" && s.StartAt != "end" && s.StartAt != "beginning" {
		return fmt.Errorf("start_position must be end or beginning")
	}
	for _, r := range s.Rules {
		if !RuleTypes[r.Type] {
			return fmt.Errorf("unknown rule type %q", r.Type)
		}
		if _, err := regexp.Compile(r.Pattern); err != nil {
			return fmt.Errorf("rule %q: %w", r.Name, err)
		}
	}
	return nil
}

// ID names a source stably across reloads, used for offsets and per-source stats.
func (s Source) ID() string {
	key := s.Type + ":" + s.Path + s.Unit + strings.Join(s.Labels, ",")
	if s.Port != 0 {
		key += fmt.Sprintf(":%d", s.Port)
	}
	return key
}

// Hash of everything the server pushed, so heartbeats can tell whether a new remote config is waiting.
func HashBytes(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:8])
}

// RemoteEnabled defaults to true: the server can only touch remote.yaml, never hand-written files.
func (a Agent) RemoteEnabled() bool { return a.RemoteConfig == nil || *a.RemoteConfig }

// Get and Set edit top-level agent.yaml keys as text, keeping comments in the file.
func Get(dir, key string) (string, error) {
	doc, err := readDoc(dir)
	if err != nil {
		return "", err
	}
	_, v := find(doc, key)
	if v == nil {
		return "", fmt.Errorf("%s is not set", key)
	}
	out, _ := yaml.Marshal(v)
	return strings.TrimSpace(string(out)), nil
}

func Set(dir, key, value string) error {
	doc, err := readDoc(dir)
	if err != nil {
		return err
	}
	var node yaml.Node
	if err := yaml.Unmarshal([]byte(value), &node); err != nil || len(node.Content) == 0 {
		node = yaml.Node{Content: []*yaml.Node{{Kind: yaml.ScalarNode, Value: value}}}
	}
	val := node.Content[0]
	root := doc.Content[0]
	if _, v := find(doc, key); v != nil {
		*v = *val
	} else {
		root.Content = append(root.Content, &yaml.Node{Kind: yaml.ScalarNode, Value: key}, val)
	}
	out, err := yaml.Marshal(doc)
	if err != nil {
		return err
	}
	path := filepath.Join(dir, "agent.yaml")
	if err := checkAgentYAML(out); err != nil {
		return err
	}
	return writeAtomic(path, out)
}

func checkAgentYAML(b []byte) error {
	var a Agent
	if err := yaml.Unmarshal(b, &a); err != nil {
		return fmt.Errorf("agent.yaml would be invalid: %w", err)
	}
	return nil
}

// CheckFile validates an edited agent.yaml or conf.d file before it replaces the original.
func CheckFile(path string, b []byte) error {
	if filepath.Base(path) == "agent.yaml" {
		return checkAgentYAML(b)
	}
	var integ Integration
	if err := yaml.Unmarshal(expand(b), &integ); err != nil {
		return err
	}
	for i, s := range integ.Logs {
		if err := s.Validate(); err != nil {
			return fmt.Errorf("logs[%d]: %w", i, err)
		}
	}
	return nil
}

func readDoc(dir string) (*yaml.Node, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "agent.yaml"))
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	doc := &yaml.Node{}
	if len(strings.TrimSpace(string(raw))) > 0 {
		if err := yaml.Unmarshal(raw, doc); err != nil {
			return nil, err
		}
	}
	if len(doc.Content) == 0 {
		doc = &yaml.Node{Kind: yaml.DocumentNode, Content: []*yaml.Node{{Kind: yaml.MappingNode}}}
	}
	return doc, nil
}

func find(doc *yaml.Node, key string) (*yaml.Node, *yaml.Node) {
	m := doc.Content[0]
	for i := 0; i+1 < len(m.Content); i += 2 {
		if m.Content[i].Value == key {
			return m.Content[i], m.Content[i+1]
		}
	}
	return nil, nil
}
