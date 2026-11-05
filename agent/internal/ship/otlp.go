package ship

// OTLP/HTTP JSON encoding of a batch (the OpenTelemetry log data model), hand-written to keep
// the agent dependency-free. Field names and value shapes follow the OTLP JSON mapping:
// lowerCamelCase keys, 64-bit integers as decimal strings, attributes as {key, value: AnyValue}.
// https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding

import (
	"path/filepath"
	"runtime"
	"strconv"
)

// OTLPPath is the standard OTLP/HTTP logs path; LegacyPath is logpipe's own format (old agents).
const (
	OTLPPath   = "/v1/logs"
	LegacyPath = "/intake/v1/logs"
	scopeName  = "logpipe-agent"
)

type kv struct {
	Key   string         `json:"key"`
	Value map[string]any `json:"value"`
}

func str(k, v string) kv { return kv{k, map[string]any{"stringValue": v}} }

func strs(k string, vs []string) kv {
	values := make([]map[string]any, len(vs))
	for i, v := range vs {
		values[i] = map[string]any{"stringValue": v}
	}
	return kv{k, map[string]any{"arrayValue": map[string]any{"values": values}}}
}

// otlpRequest groups events by resource (host, service, source, UTC offset) into an
// ExportLogsServiceRequest. The agent doesn't parse a line's own time, so only
// observedTimeUnixNano is set; the server reads the event time from the line.
func otlpRequest(batch []Event, version string) map[string]any {
	type key struct{ host, service, source, offset string }
	order := []key{}
	records := map[key][]map[string]any{}
	for _, e := range batch {
		k := key{e.Host, e.Service, e.Source, e.Time.Format("-07:00")}
		if _, ok := records[k]; !ok {
			order = append(order, k)
		}
		attrs := []kv{}
		if e.Path != "" {
			attrs = append(attrs, str("log.file.path", e.Path), str("log.file.name", filepath.Base(e.Path)))
		}
		if len(e.Tags) > 0 {
			attrs = append(attrs, strs("logpipe.tags", e.Tags))
		}
		records[k] = append(records[k], map[string]any{
			"observedTimeUnixNano": strconv.FormatInt(e.Time.UnixNano(), 10),
			"body":                 map[string]any{"stringValue": e.Message},
			"attributes":           attrs,
		})
	}
	resources := make([]map[string]any, 0, len(order))
	for _, k := range order {
		resources = append(resources, map[string]any{
			"resource": map[string]any{"attributes": []kv{
				str("host.name", k.host), str("service.name", k.service),
				str("os.type", runtime.GOOS), str("host.arch", runtime.GOARCH),
				// Not in the semantic conventions: the host's offset, so zone-less times in a line
				// are read as that host's wall clock.
				str("logpipe.source", k.source), str("logpipe.host.utc_offset", k.offset),
			}},
			"scopeLogs": []map[string]any{{
				"scope":      map[string]any{"name": scopeName, "version": version},
				"logRecords": records[k],
			}},
		})
	}
	return map[string]any{"resourceLogs": resources}
}

func countRecords(req map[string]any) int {
	n := 0
	for _, r := range asList(req["resourceLogs"]) {
		for _, s := range asList(asMap(r)["scopeLogs"]) {
			n += len(asList(asMap(s)["logRecords"]))
		}
	}
	return n
}

func asList(v any) []any {
	l, _ := v.([]any)
	return l
}

func asMap(v any) map[string]any {
	m, _ := v.(map[string]any)
	return m
}
