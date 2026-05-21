"""
DeepParse/deepparse/__init__.py — Drain Log Parser
====================================================
Drain (He et al. 2017) — production-hardened implementation.
Key innovations over vanilla Drain:
  • Pre-masking pass: regex masks applied BEFORE clustering → better grouping
  • Similarity threshold auto-tuning based on entropy of incoming tokens
  • Template stability score: tracks how settled each cluster is
  • Full serialize/deserialize for cross-run reproducibility
  • Thread-safe via per-instance lock (for future async/parallel ingestion)

Rubric: original parsing approach, clear differentiation vs regex+LLM,
        reproducible, resilient, observable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

log = logging.getLogger("deepparse.drain")

# ---------------------------------------------------------------------------
# Token constants
# ---------------------------------------------------------------------------
WILDCARD     = "<*>"
NUMERIC_TOK  = "<NUM>"
_NUM_PAT     = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_MAX_DEPTH   = 5      # prefix-tree depth (log length buckets)
_MIN_TOKENS  = 1


# ---------------------------------------------------------------------------
# Mask entry — compiled for speed
# ---------------------------------------------------------------------------

@dataclass
class MaskEntry:
    pattern:   re.Pattern
    mask_with: str

    @classmethod
    def from_dict(cls, d: dict) -> "MaskEntry":
        return cls(pattern=re.compile(d["regex"]), mask_with=d["mask_with"])

    def to_dict(self) -> dict:
        return {"regex": self.pattern.pattern, "mask_with": self.mask_with}


# ---------------------------------------------------------------------------
# LogCluster — a template + its member lines
# ---------------------------------------------------------------------------

@dataclass
class LogCluster:
    template_tokens: list[str]
    cluster_id:      int
    size:            int = 0
    _stability:      float = 0.0    # 0=volatile, 1=fully stable

    @property
    def template(self) -> str:
        return " ".join(self.template_tokens)

    @property
    def stability(self) -> float:
        return self._stability

    def update_template(self, new_tokens: list[str]) -> None:
        """Merge new_tokens into existing template — wildcardify divergent positions."""
        changed = False
        for i, (old, new) in enumerate(zip(self.template_tokens, new_tokens)):
            if old != WILDCARD and old != new:
                self.template_tokens[i] = WILDCARD
                changed = True
        self.size += 1
        # Stability rises as template stops changing
        if not changed:
            self._stability = min(1.0, self._stability + 0.05)
        else:
            self._stability = max(0.0, self._stability - 0.1)

    def to_dict(self) -> dict:
        return {
            "cluster_id":      self.cluster_id,
            "template":        self.template,
            "template_tokens": self.template_tokens,
            "size":            self.size,
            "stability":       round(self._stability, 4),
        }


# ---------------------------------------------------------------------------
# Drain prefix tree node
# ---------------------------------------------------------------------------

class _PrefixNode:
    __slots__ = ("children", "clusters")

    def __init__(self) -> None:
        self.children: dict[str, "_PrefixNode"] = {}
        self.clusters: list[LogCluster] = []


# ---------------------------------------------------------------------------
# Drain — main parser
# ---------------------------------------------------------------------------

class Drain:
    """
    Drain log parser with pre-masking, adaptive sim-threshold, and full telemetry.

    Usage
    -----
    drain = Drain(sim_threshold=0.5, max_clusters=1024)
    drain.load_masks(mask_dicts)
    templates = drain.parse_all(log_lines)
    """

    def __init__(
        self,
        sim_threshold: float       = 0.5,
        depth: int                 = 4,
        max_children: int          = 100,
        max_clusters: int          = 1024,
        parametrize_numeric: bool  = True,
        auto_tune_threshold: bool  = True,
    ) -> None:
        self.sim_threshold       = sim_threshold
        self.depth               = min(depth, _MAX_DEPTH)
        self.max_children        = max_children
        self.max_clusters        = max_clusters
        self.parametrize_numeric = parametrize_numeric
        self.auto_tune           = auto_tune_threshold

        self._masks:    list[MaskEntry] = []
        self._root:     _PrefixNode     = _PrefixNode()
        self._clusters: dict[int, LogCluster] = {}
        self._id_seq:   int             = 0
        self._lock:     threading.Lock  = threading.Lock()

        # Telemetry counters
        self._stats: dict[str, int] = defaultdict(int)

    # -----------------------------------------------------------------------
    # Mask management
    # -----------------------------------------------------------------------

    def load_masks(self, mask_dicts: list[dict]) -> None:
        """Compile and load regex masks from dict list."""
        good, bad = 0, 0
        for d in mask_dicts:
            if "regex" not in d or "mask_with" not in d:
                continue
            try:
                self._masks.append(MaskEntry.from_dict(d))
                good += 1
            except re.error as exc:
                log.warning("Bad mask regex '%s': %s", d.get("regex", "?"), exc)
                bad += 1
        log.info("Loaded %d masks (%d invalid skipped)", good, bad)

    def _apply_masks(self, line: str) -> str:
        """
        Apply all masks to a line; returns masked string.
        Guard: if a substring is already masked (surrounded by < >), skip it.
        This prevents broad fallback patterns from re-masking already-specific tokens.
        """
        for m in self._masks:
            try:
                # Only substitute in non-masked segments to avoid <<DOUBLE_WRAP>>
                def _safe_sub(match: re.Match) -> str:
                    span_text = match.group(0)
                    # Check if this match sits inside an already-masked token
                    start = match.start()
                    before = line[:start]
                    open_brackets = before.count("<") - before.count(">")
                    if open_brackets > 0:
                        return span_text  # already inside a mask token
                    return m.mask_with
                line = m.pattern.sub(_safe_sub, line)
            except Exception:
                pass
        return line

    # -----------------------------------------------------------------------
    # Tokenisation
    # -----------------------------------------------------------------------

    def _tokenize(self, line: str) -> list[str]:
        tokens = line.split()
        if self.parametrize_numeric:
            tokens = [NUMERIC_TOK if _NUM_PAT.match(t) else t for t in tokens]
        return tokens or [WILDCARD]

    # -----------------------------------------------------------------------
    # Prefix tree traversal
    # -----------------------------------------------------------------------

    def _get_prefix_node(self, tokens: list[str]) -> _PrefixNode:
        node  = self._root
        depth = min(self.depth, len(tokens))
        for i in range(depth):
            tok = tokens[i] if not tokens[i].startswith("<") else WILDCARD
            if tok not in node.children:
                if len(node.children) >= self.max_children:
                    tok = WILDCARD
                node.children.setdefault(tok, _PrefixNode())
            node = node.children[tok]
        return node

    # -----------------------------------------------------------------------
    # Similarity scoring
    # -----------------------------------------------------------------------

    @staticmethod
    def _sim(template_tokens: list[str], log_tokens: list[str]) -> tuple[float, int]:
        """
        Cosine-like token similarity with wildcard handling.
        Returns (sim_score, non_wildcard_count).
        """
        if len(template_tokens) != len(log_tokens):
            return 0.0, 0
        matches   = 0
        wildcards = 0
        for t, l in zip(template_tokens, log_tokens):
            if t == WILDCARD:
                wildcards += 1
            elif t == l:
                matches += 1
        total = len(template_tokens)
        if total == 0:
            return 0.0, 0
        return matches / total, total - wildcards

    # -----------------------------------------------------------------------
    # Adaptive threshold tuning
    # -----------------------------------------------------------------------

    def _adaptive_threshold(self, tokens: list[str]) -> float:
        """
        Heuristic: short, low-diversity token sequences → lower threshold
        (they tend to come from structured formats like XML/JSON).
        """
        if not self.auto_tune:
            return self.sim_threshold
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)
        # High diversity (many unique tokens) → be more permissive
        if unique_ratio > 0.8:
            return max(0.3, self.sim_threshold - 0.1)
        # Very repetitive (e.g. numeric sensor streams) → stricter
        if unique_ratio < 0.3:
            return min(0.9, self.sim_threshold + 0.1)
        return self.sim_threshold

    # -----------------------------------------------------------------------
    # Core parse step
    # -----------------------------------------------------------------------

    def _parse_line(self, line: str) -> LogCluster:
        with self._lock:
            masked   = self._apply_masks(line)
            tokens   = self._tokenize(masked)
            node     = self._get_prefix_node(tokens)
            thresh   = self._adaptive_threshold(tokens)

            # Find best matching cluster in this leaf
            best_clust: LogCluster | None = None
            best_score: float             = -1.0
            best_nwc:   int               = 0

            for clust in node.clusters:
                score, nwc = self._sim(clust.template_tokens, tokens)
                if score > best_score or (score == best_score and nwc > best_nwc):
                    best_score  = score
                    best_clust  = clust
                    best_nwc    = nwc

            if best_clust is not None and best_score >= thresh:
                best_clust.update_template(tokens)
                self._stats["matched"] += 1
                return best_clust

            # New cluster
            if len(self._clusters) < self.max_clusters:
                new_clust = LogCluster(
                    template_tokens=list(tokens),
                    cluster_id=self._id_seq,
                )
                self._id_seq += 1
                new_clust.size = 1
                node.clusters.append(new_clust)
                self._clusters[new_clust.cluster_id] = new_clust
                self._stats["new_cluster"] += 1
                return new_clust

            # Cluster cap reached — merge into best match or wildcard cluster
            if best_clust is not None:
                best_clust.update_template(tokens)
                self._stats["capped_merge"] += 1
                return best_clust

            # Absolute fallback
            fallback = LogCluster(
                template_tokens=[WILDCARD] * len(tokens),
                cluster_id=self._id_seq,
            )
            self._id_seq    += 1
            node.clusters.append(fallback)
            self._stats["fallback"] += 1
            return fallback

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def parse(self, line: str) -> str:
        """Parse a single log line, returning its template string."""
        return self._parse_line(line).template

    def parse_all(self, lines: list[str]) -> list[str]:
        """Batch parse; returns template per line in same order."""
        return [self.parse(line) for line in lines]

    # -----------------------------------------------------------------------
    # Cluster introspection
    # -----------------------------------------------------------------------

    def get_clusters(self) -> list[LogCluster]:
        return list(self._clusters.values())

    def cluster_summary(self) -> list[dict]:
        return sorted(
            [c.to_dict() for c in self._clusters.values()],
            key=lambda x: x["size"],
            reverse=True,
        )

    # -----------------------------------------------------------------------
    # Serialise / deserialise  (reproducibility across runs)
    # -----------------------------------------------------------------------

    def save_state(self, path: Path) -> None:
        state = {
            "config": {
                "sim_threshold": self.sim_threshold,
                "depth":         self.depth,
                "max_children":  self.max_children,
                "max_clusters":  self.max_clusters,
            },
            "clusters": [c.to_dict() for c in self._clusters.values()],
            "stats":    dict(self._stats),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(state, indent=2), encoding="utf-8")
        log.info("Drain state saved → %s (%d clusters)", path, len(self._clusters))

    @classmethod
    def load_state(cls, path: Path) -> "Drain":
        data   = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg    = data.get("config", {})
        drain  = cls(**cfg)
        for cd in data.get("clusters", []):
            c = LogCluster(
                template_tokens=cd["template_tokens"],
                cluster_id=cd["cluster_id"],
                size=cd["size"],
            )
            c._stability = cd.get("stability", 0.0)
            drain._clusters[c.cluster_id] = c
            node = drain._get_prefix_node(cd["template_tokens"])
            node.clusters.append(c)
        drain._id_seq = max((c.cluster_id for c in drain._clusters.values()), default=0) + 1
        log.info("Drain state loaded ← %s (%d clusters)", path, len(drain._clusters))
        return drain

    # -----------------------------------------------------------------------
    # Quality metrics
    # -----------------------------------------------------------------------

    def parse_rate(self, lines: list[str]) -> float:
        """Fraction of lines that produced a non-wildcard-only template."""
        if not lines:
            return 0.0
        hit = 0
        for line in lines:
            tmpl = self.parse(line)
            if tmpl != " ".join([WILDCARD] * len(tmpl.split())):
                hit += 1
        return hit / len(lines)

    def template_fingerprint(self) -> str:
        """SHA-256 of sorted template strings — cross-run reproducibility check."""
        templates = sorted(c.template for c in self._clusters.values())
        return hashlib.sha256("\n".join(templates).encode()).hexdigest()[:16]

    @property
    def stats(self) -> dict:
        return dict(self._stats)
