"""The template index: one vector per template, two ways.

* **text** (question -> template): a sentence-embedding model (``LOGPIPE_ASK_MODEL``, by default
  ``all-MiniLM-L6-v2``) over each template's words, with identifiers split first
  (``OutOfMemoryError`` -> out of memory error) so the model reads words, not one long token.
* **LogBERT** (template -> templates that occur in the same places): each stream model's token
  embedding row for the template, averaged over the streams that know it. These are learned
  from sequences, not text, so they only serve "neighbours of the anchor event".

Templates number in the hundreds, so search is a brute-force cosine over a numpy matrix, rebuilt
when the registry changes (``AskService.index_for``); swapping in a vector database means
implementing ``search`` and ``neighbours``.

The shutdown set (templates that say a process stopped or can't go on: OOM, SIGTERM, segfault,
connection refused, disk full, ...) is a fixed pattern plus the templates closest to a few seed
phrases, and a person can add or remove templates (control doc ``ask_shutdown``).
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..settings import get_settings

SHUTDOWN_PHRASES = (
    "out of memory",
    "killed process",
    "received shutdown request",
    "terminated by signal",
    "segmentation fault",
    "connection refused",
    "no space left on device",
    "panic",
    "fatal error",
    "exited with code",
    "shutting down",
    "unknown directive emerg",
)
SHUTDOWN_RX = re.compile(
    r"(?i)out ?of ?memory|oom[- ]?kill|killed process|\bsig(term|kill|segv|abrt)\b|signal (6|9|11|15)\b|"
    r"segmentation fault|segfault|core dumped|shut(ting)? ?down|shutdown request|\bterminat(ed|ing)\b|"
    r"\bpanic\b|\bfatal\b|connection refused|\brefused\b|no space left|enospc|disk (is )?full|"
    r"exit(ed)? (with )?(code|status)|\bcrash|\[emerg\]|aborting"
)


def readable(text: str) -> str:
    """A template as words: slots named or dropped, camelCase and dotted.identifiers split."""
    text = re.sub(r"<\*(?::(\w+))?>", lambda m: f" {m.group(1) or ''} ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return " ".join(re.findall(r"[A-Za-z]+|\d{3}", text))


_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def text_model(name: str):
    """The sentence model, loaded once per process (a ``sentence-transformers`` dependency)."""
    from sentence_transformers import SentenceTransformer

    with _LOCK:
        return SentenceTransformer(name)


def _encode(model, texts: list[str]) -> np.ndarray:
    return np.asarray(
        model.encode([readable(t) or t for t in texts], normalize_embeddings=True), dtype=np.float32
    )


@dataclass
class TemplateIndex:
    version: int
    ids: list[int]
    texts: list[str]
    text_vectors: np.ndarray
    model: object
    logbert: dict[int, np.ndarray] = field(default_factory=dict)
    kind: str = "minilm"

    @classmethod
    def build(
        cls, templates: dict[int, str], version: int, scorers: dict | None = None, snapshot=None
    ) -> TemplateIndex:
        model = text_model(get_settings().ask.model)
        ids, texts = list(templates), list(templates.values())
        vectors = _encode(model, texts) if ids else np.zeros((0, 0), np.float32)
        return cls(version, ids, texts, vectors, model, _logbert_vectors(scorers or {}, snapshot))

    def search(self, text: str, k: int = 5) -> list[tuple[int, float]]:
        if not self.ids or not text.strip():
            return []
        sims = self.text_vectors @ _encode(self.model, [text])[0]
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], round(float(sims[i]), 3)) for i in order if sims[i] > 0]

    def neighbours(self, tid: int, k: int = 5) -> list[tuple[int, float]]:
        """Templates the LogBERT models place near ``tid`` (seen in similar contexts)."""
        if tid not in self.logbert:
            return []
        me = self.logbert[tid]
        sims = [(t, float(v @ me)) for t, v in self.logbert.items() if t != tid]
        return [(t, round(s, 3)) for t, s in sorted(sims, key=lambda x: -x[1])[:k]]


def _logbert_vectors(scorers: dict, snapshot) -> dict[int, np.ndarray]:
    sums: dict[int, list[np.ndarray]] = {}
    for scorer in scorers.values():
        table = scorer.model.embedding.token.weight.detach().cpu().numpy()
        for tid in scorer.vocab.template_ids:
            sums.setdefault(snapshot.canonical_id(tid) if snapshot else tid, []).append(
                table[scorer.vocab.index(tid)]
            )
    out = {}
    for tid, vs in sums.items():
        v = np.mean(vs, axis=0)
        n = float(np.linalg.norm(v))
        out[tid] = v / n if n else v
    return out


def auto_shutdown(templates: dict[int, str], index: TemplateIndex | None) -> dict[int, str]:
    """Shutdown-type templates and why each was picked."""
    found: dict[int, str] = {}
    for tid, text in templates.items():
        if m := SHUTDOWN_RX.search(text):
            found[tid] = f"says {m.group(0)!r}"
    if index is not None:
        floor = get_settings().ask.shutdown_min_score
        for phrase in SHUTDOWN_PHRASES:
            for tid, score in index.search(phrase, 10):
                if score >= floor and tid not in found:
                    found[tid] = f"close to {phrase!r} ({score:.2f})"
    return found


def shutdown_set(control, templates: dict[int, str], index: TemplateIndex | None) -> set[int]:
    edits = control.get("ask_shutdown", "edits") or {}
    auto = set(auto_shutdown(templates, index))
    return (auto | set(edits.get("add", []))) - set(edits.get("remove", []))
