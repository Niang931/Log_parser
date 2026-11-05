// Thin typed client for `logpipe server`'s /api.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

export class ApiError extends Error {}

export async function api<T>(path: string, init?: RequestInit & { json?: unknown }): Promise<T> {
  const { json, ...rest } = init ?? {};
  const res = await fetch(path, {
    ...rest,
    headers: json !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const detail = data?.detail;
    throw new ApiError(typeof detail === "string" ? detail : `${res.status} ${res.statusText}`);
  }
  return data as T;
}

/** Polls every `every` ms (default 2s), like the live parts of the UI need. */
export function usePoll<T>(path: string | null, every = 2000) {
  return useQuery<T>({
    queryKey: [path],
    queryFn: () => api<T>(path!),
    enabled: path !== null,
    refetchInterval: every,
  });
}

/** Refetches fleet views the moment the server reports a change (agent up/down, heartbeat,
 * pushed config) over server-sent events; polling stays as the fallback. */
export function useServerEvents() {
  const qc = useQueryClient();
  useEffect(() => {
    const es = new EventSource("/api/events");
    es.addEventListener("change", () =>
      qc.invalidateQueries({ predicate: (q) => /^\/api\/(agents|overview|setup|review|maintenance|jobs)/.test(String(q.queryKey[0])) }));
    return () => es.close();
  }, [qc]);
}

/** A POST/PUT/DELETE that refreshes every query afterwards (the UI is small; that's cheap). */
export function useAction<A = void, R = unknown>(fn: (arg: A) => Promise<R>) {
  const qc = useQueryClient();
  return useMutation<R, Error, A>({ mutationFn: fn, onSettled: () => qc.invalidateQueries() });
}

export type Status = "FINAL" | "PENDING";

export interface Settings {
  sample_cap: number;
  retrain_days: number;
  calibrated: boolean;
  calibrated_at: string | null;
  setup_done: boolean;
  draft_lines: number;
  draft_after_minutes: number;
}

export interface Job {
  id: string;
  kind: "draft" | "apply" | "retrain" | "calibrate" | "recalibrate" | "explain";
  params: { sources?: string[]; streams?: string[]; batch?: string; incident?: string };
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  progress: number;
  message: string;
  error?: string;
  log?: [string, string][];
  /** section -> what the job took in, the LLM prompts it sent, and what it returned */
  detail?: Record<string, unknown>;
  created_at: string;
  started_at?: string;
  finished_at?: string;
}

/** have/need for something that only starts once there is enough data. */
export interface Readiness {
  have: number;
  need: number;
  at?: string;
}

export interface Batch {
  id: string;
  initial: boolean;
  sources: string[];
  status: "review" | "applied" | "discarded" | "replaced";
  lines: number;
  sampled?: number;
  report: { holdout_coverage?: number };
  created_at: string;
  drafts: number | Draft[];
  pending?: number;
  approved?: number;
  note?: string;
}

export interface Draft {
  id: string;
  batch: string;
  tokens: string[];
  keys: string[];
  count: number;
  samples: string[];
  status: "pending" | "approved" | "rejected" | "merged";
  edited?: boolean;
  merged_into?: string;
  rule: string;
  rule_edited: boolean;
}

export interface AutoTemplate {
  template_id: number;
  tokens: string[];
  keys: string[];
  rule: string;
  rule_edited: boolean;
  count: number;
  first_seen: string | null;
}

export interface Collecting {
  id: string;
  host: string;
  service: string;
  source: string;
  have: number;
  need: number;
  first_seen: string;
  draft_by: string;
}

export interface Maintenance {
  review: Batch[];
  auto_templates: number;
  collecting: Collecting[];
  waiting: Stream[];
  warnings: (Warning & { readiness?: ({ stream: string; name: string } & Partial<Readiness>)[] })[];
}

export interface Alert {
  alert_id: string;
  key: string;
  window_start: string;
  window_end: string;
  template_ids: number[];
  miss_ratio: number;
  novelty_ratio: number;
  reasons: string[];
  model_version: string;
}

export type IncidentStatus = "open" | "acknowledged" | "resolved" | "false_positive";

/** One service's anomaly alerts close together in time. */
export interface Incident {
  id: string;
  key: string;
  host: string;
  service: string;
  start: string;
  end: string;
  alerts: number;
  peak_miss: number;
  model_version: string;
  stream: string;
  status: IncidentStatus;
  note: string;
  updated_at?: string | null;
  explanation?: { summary: string; likely_cause: string; next_steps: string[]; at: string; note?: string } | null;
}

export interface IncidentEvent {
  timestamp: string;
  host: string;
  service: string;
  message: string;
  level: string;
  template_id: number;
  template: string;
  /** expected: the model predicted it; unexpected: it predicted something else; unseen: a kind of line it never learned */
  status: "expected" | "unexpected" | "unseen";
  /** what the model expected there instead; ``words`` is the template without its variable slots */
  expected: { template_id: number; template: string; words: string }[];
}

export interface IncidentDetail extends Incident {
  events: IncidentEvent[];
  around: { timestamp: string; host: string; service: string; message: string; level: string }[];
  model_current: boolean;
  explain_job: Job | null;
}

export interface Overview {
  received: number;
  rate: number;
  status: Partial<Record<Status, number>>;
  inbox: number;
  agents: number;
  agents_online: number;
  sources: number;
  templates: number;
  registry_version: number;
  warnings: Record<string, number>;
  review: number;
  streams: number;
  models: number;
  settings: Settings;
  jobs: Job[];
  recent_alerts: Alert[];
}

export type Level = "error" | "warn" | "info" | "debug";

export interface Histogram {
  step: number;
  points: ({ t: number } & Record<string, number>)[];
}

export type Facets = Record<"level" | "service" | "host" | "source" | "status", [string, number][]>;

export interface LogLine {
  log_id: string;
  timestamp: string;
  host: string;
  service: string;
  source: string;
  tags: string[];
  status: Status;
  template_id: number;
  payload: Record<string, string>;
  message: string;
  level: Level;
}

export interface Template {
  template_id: number;
  tokens: string[];
  keys: string[];
  alias_of: number | null;
  text: string;
  count: number;
  first_seen: string | null;
  last_seen: string | null;
  seen_by: string[];
  candidate: boolean;
  sql: string;
  rule: string;
  rule_edited: boolean;
  linked_by: "you" | "llm" | "pipeline" | null;
  link_reason?: string | null;
  canonical: number;
  linked: number[];
}

export interface Source {
  id: string;
  host: string;
  service: string;
  source: string;
  first_seen: string;
  last_seen?: string;
  count: number;
  calibrated: boolean;
  alias_of: string | null;
  streams: string[];
}

export interface Selector {
  host?: string;
  service?: string;
  source?: string;
}

export interface Stream {
  id: string;
  name: string;
  /** sources ticked by hand; a source is in exactly one stream */
  picked?: string[];
  /** patterns that also catch future sources (advanced) */
  selectors: Selector[];
  retrain_days: number | null;
  created_at: string;
  trained_at?: string;
  model_version?: string;
  windows?: number;
  miss_threshold?: number;
  model_error?: string | null;
  waiting?: boolean;
  data?: Readiness;
  sources: string[];
  has_model: boolean;
  vocab: number;
}

export interface Warning {
  id: string;
  kind: "new_source" | "novel_template";
  subject: string;
  title: string;
  detail: string;
  first_seen: string;
  last_seen?: string;
  count: number;
  status: "open" | "resolved";
  streams: string[];
  resolution?: string;
  resolved_at?: string;
}

export interface AgentSource {
  id: string;
  type: string;
  target: string;
  service: string;
  lines: number;
  bytes: number;
  last_line_at?: string;
  files?: string[];
  error?: string;
}

export interface AgentConfigSource {
  type: string;
  path?: string;
  port?: number;
  unit?: string;
  service?: string;
  source?: string;
  tags?: string[];
  log_processing_rules?: { type: string; name?: string; pattern: string }[];
  file: string;
}

export interface AgentStatus {
  hostname: string;
  version: string;
  os: string;
  arch: string;
  started_at: string;
  endpoint: string;
  tags: string[] | null;
  sources: AgentSource[] | null;
  shipper: { sent: number; dropped: number; buffer_bytes: number; last_error?: string; last_send_at?: string };
  config: AgentConfigSource[] | null;
  remote_hash: string;
  config_error?: string;
}

export interface Agent {
  hostname: string;
  first_seen: string;
  last_seen: string;
  key_name: string;
  online: boolean;
  status: AgentStatus;
  remote: { yaml?: string; hash?: string; updated_at?: string };
  remote_applied: boolean;
}

export interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  revoked: boolean;
  last_used_at: string | null;
}

export interface Stats {
  hosts: [string, number][];
  services: [string, number][];
  sources: [string, number][];
  templates: [number, number, string][];
}

export interface QueryResult {
  columns: string[];
  rows: unknown[][];
  engine: string;
  table: string;
}
