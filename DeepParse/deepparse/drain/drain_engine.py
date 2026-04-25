"""Deterministic Drain-style parser with typed-placeholder masks.

This module is an extended port of the open source ``Drain3`` algorithm
that follows the *Mask-First* strategy from the DeepParse paper.  Masks
are applied first to substitute typed placeholders (e.g. ``<VAR:IP>``);
then a fixed-depth parse tree clusters the remaining tokens.  Tokens
that vary across logs grouped into the same template are replaced with
the wildcard ``<*>``.

Properties guaranteed by this implementation:

* ``parse`` is ``O(N)`` for ``N`` log lines (depth and similarity
  threshold are bounded constants).
* ``parse`` is deterministic — calling it twice on the same input
  produces identical templates and identical cluster ids.
* Identical log lines always receive the same template id (matches the
  guarantee called out in the paper, Section "Integration with Drain").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from ..masks_types import Mask
from ..tokenize import tokenize
from .masks_application import MaskApplier

WILDCARD = "<*>"
_VAR_PLACEHOLDER_RE = re.compile(r"<VAR:[A-Z0-9]+>")


def _is_placeholder(token: str) -> bool:
    """Return True for typed mask placeholders such as ``<VAR:IP>``."""
    return token.startswith("<VAR:") and token.endswith(">")


def _to_wildcard(token: str) -> str:
    """Normalise any typed placeholders inside ``token`` to ``<*>``.

    Handles both bare placeholders (``<VAR:IP>``) and composite tokens
    that embed a placeholder (``Worker-<VAR:NUMBER>``).
    """
    return _VAR_PLACEHOLDER_RE.sub(WILDCARD, token)


@dataclass
class DrainCluster:
    cluster_id: int
    template: List[str]
    size: int = 0

    def similarity(self, tokens: Sequence[str]) -> float:
        if not self.template:
            return 0.0
        matches = 0
        for tmpl_tok, tok in zip(self.template, tokens):
            if tmpl_tok == tok or tmpl_tok == WILDCARD:
                matches += 1
        return matches / len(self.template)

    def update(self, tokens: Sequence[str]) -> None:
        self.size += 1
        if not self.template:
            self.template = list(tokens)
            return
        # Same length is required for clustering; merge by tokenwise
        # generalisation.  Typed placeholders survive merges; identical
        # literal tokens are kept; everything else collapses to <*>.
        for idx, tok in enumerate(tokens):
            current = self.template[idx]
            if current == tok:
                continue
            self.template[idx] = WILDCARD

    def template_str(self) -> str:
        """Render the cluster template as a string.

        Typed placeholders such as ``<VAR:IP>`` are normalised to the
        LogHub-canonical wildcard ``<*>`` so that templates can be
        compared directly against LogHub-2k ground truth.  Internally
        the parse tree continues to key on typed placeholders; the
        normalisation only affects the rendered string.
        """
        return " ".join(_to_wildcard(tok) for tok in self.template)

    def template_str_typed(self) -> str:
        """Render the cluster template preserving typed placeholders."""
        return " ".join(self.template)


@dataclass
class DrainEngine:
    """Drain-style log parser with mask-first preprocessing.

    Defaults match the Drain3 baseline configuration cited in the paper
    (Section "Baseline Parsers"): ``depth=5``, ``similarity_threshold=0.4``,
    ``max_children=100``.
    """

    depth: int = 5
    similarity_threshold: float = 0.4
    max_children: int = 100
    masks: Sequence[Mask] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.applier = MaskApplier(self.masks)
        # Bucket clusters by length and a depth-bounded prefix so that
        # lookup is O(1) amortised.  Tokens of any kind that exceed the
        # bucket's depth do not affect grouping.
        self._buckets: Dict[Tuple[int, str], List[DrainCluster]] = {}
        self._next_id: int = 0

    # ---- internal helpers --------------------------------------------------
    def _cluster_key(self, tokens: Sequence[str]) -> Tuple[int, str]:
        prefix_tokens: List[str] = []
        for tok in tokens[: self.depth]:
            # Treat typed placeholders and pure-digit tokens as the same
            # "bucket key" so messages with different concrete values
            # cluster together at the prefix layer.
            if _is_placeholder(tok) or tok.isdigit():
                prefix_tokens.append(WILDCARD)
            else:
                prefix_tokens.append(tok)
        return len(tokens), " ".join(prefix_tokens)

    def _allocate_id(self) -> int:
        cid = self._next_id
        self._next_id += 1
        return cid

    # ---- public API --------------------------------------------------------
    def add_log(self, line: str) -> DrainCluster:
        masked_line = self.applier.apply(line)
        tokens = tokenize(masked_line)
        if not tokens:
            tokens = [""]
        key = self._cluster_key(tokens)
        cluster_list = self._buckets.setdefault(key, [])

        best_cluster: DrainCluster | None = None
        best_score = -1.0
        for cluster in cluster_list:
            score = cluster.similarity(tokens)
            if score > best_score:
                best_score = score
                best_cluster = cluster

        if best_cluster is not None and best_score >= self.similarity_threshold:
            best_cluster.update(tokens)
            return best_cluster

        # Create a new cluster.  Bound the number of clusters per bucket
        # to keep the parse tree shallow on adversarial inputs.
        if len(cluster_list) >= self.max_children and cluster_list:
            fallback = cluster_list[0]
            fallback.update(tokens)
            return fallback
        new_cluster = DrainCluster(
            cluster_id=self._allocate_id(),
            template=list(tokens),
            size=0,
        )
        new_cluster.update(tokens)
        cluster_list.append(new_cluster)
        return new_cluster

    def parse(self, lines: Iterable[str]) -> List[str]:
        """Parse a sequence of lines, returning *final* template strings.

        Drain mutates a cluster's template each time a new line is added
        to it; we therefore make two passes so that every returned
        template reflects the cluster's converged state.
        """
        return [template for _cid, template in self.parse_with_ids(lines)]

    def parse_with_ids(self, lines: Iterable[str]) -> List[Tuple[int, str]]:
        """Parse a sequence of lines and return ``(cluster_id, template)`` pairs.

        Two-pass: first add every line so the cluster templates converge,
        then look up the final template for each line by cluster id.

        .. warning::
           The returned templates are a *snapshot* taken at the end of
           this call.  If you subsequently call :meth:`add_log` (or
           :meth:`parse_with_ids` / :meth:`parse` again with new lines),
           the cluster templates may merge further and the strings
           previously returned will no longer reflect the engine's
           current state.  For the paper-aligned usage (Listing 1:
           ``Drain().load_masks(p).parse_all(logs)``) this is irrelevant
           because the entire corpus is processed in a single call.
        """
        line_list = list(lines)
        cluster_ids: List[int] = [self.add_log(line).cluster_id for line in line_list]
        templates_by_id: Dict[int, str] = {}
        for clusters in self._buckets.values():
            for cluster in clusters:
                templates_by_id[cluster.cluster_id] = cluster.template_str()
        return [(cid, templates_by_id[cid]) for cid in cluster_ids]

    @property
    def num_clusters(self) -> int:
        return sum(len(v) for v in self._buckets.values())
