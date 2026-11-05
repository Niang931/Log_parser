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
      qc.invalidateQueries({ predicate: (q) => /^\/api\/(agents|overview|setup)/.test(String(q.queryKey[0])) }));
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
}

export interface Job {
  id: string;
  kind: "calibrate" | "recalibrate" | "retrain";
  params: Record<string, unknown>;
  status: "queued" | "running" | "done" | "failed";
  progress: number;
  message: string;
  error?: string;
  created_at: string;
  started_at?: string;
  finished_at?: string;
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
  streams: number;
  models: number;
  settings: Settings;
  jobs: Job[];
  recent_alerts: Alert[];
}

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
  selectors: Selector[];
  retrain_days: number | null;
  created_at: string;
  trained_at?: string;
  model_version?: string;
  windows?: number;
  miss_threshold?: number;
  model_error?: string | null;
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
  per_minute: { minute: string; FINAL: number; PENDING: number }[];
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
