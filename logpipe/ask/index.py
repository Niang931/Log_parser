"""The template index: one vector per template, two ways.

* **text** (question -> template): a sentence-embedding model (``all-MiniLM-L6-v2``) when the
  ``ask`` extra is installed, else TF-IDF over the template's words, its split identifiers
  (``OutOfMemoryError`` -> out of memory error) and a small synonym table, so "out of memory"
  finds ``OutOfMemoryError`` and "disk full" finds ``No space left on device``.
* **LogBERT** (template -> templates that occur in the same places): each stream model's token
  embedding row for the template, averaged over the streams that know it. These are learned
  from sequences, not text, so they only serve "neighbours of the anchor event".

Templates number in the hundreds, so search is a brute-force cosine over a numpy matrix, rebuilt
when the registry version changes (``index_for``); swapping in a vector database means
implementing ``search`` and ``neighbours``.

The shutdown set (templates that say a process stopped or can't go on: OOM, SIGTERM, segfault,
connection refused, disk full, ...) is found once per registry version from seed phrases, and a
person can add or remove templates (control doc ``ask_shutdown``).
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

SYNONYMS = {
    "oom": "out of memory", "outofmemory": "out of memory", "enospc": "no space left on device disk full",
    "sigterm": "signal terminate shutdown", "sigkill": "signal killed", "sigsegv": "segmentation fault",
    "segfault": "segmentation fault", "econnrefused": "connection refused", "refused": "refused connection",
    "full": "no space left", "disk": "space device", "killed": "kill terminated", "shutdown": "shutting down stop",
    "crash": "crashed terminated fatal", "timeout": "timed out", "dns": "unknownhost name service not known",
    "tls": "ssl certificate handshake", "cert": "certificate", "502": "bad gateway", "503": "service unavailable",
    "memory": "heap", "heap": "memory", "restart": "starting shutdown", "down": "shutdown refused unavailable",
}
SHUTDOWN_PHRASES = ("out of memory", "killed process", "received shutdown request", "terminated by signal",
                    "segmentation fault", "connection refused", "no space left on device", "panic", "fatal error",
                    "exited with code", "shutting down", "unknown directive emerg")
SHUTDOWN_RX = re.compile(r"(?i)out ?of ?memory|oom[- ]?kill|killed process|\bsig(term|kill|segv|abrt)\b|signal (6|9|11|15)\b|"
                         r"segmentation fault|segfault|core dumped|shut(ting)? ?down|shutdown request|\bterminat(ed|ing)\b|"
                         r"\bpanic\b|\bfatal\b|connection refused|\brefused\b|no space left|enospc|disk (is )?full|"
                         r"exit(ed)? (with )?(code|status)|\bcrash|\[emerg\]|aborting")
SHUTDOWN_MIN_SCORE = 0.45


def split_words(text: str) -> list[str]:
    text = re.sub(r"<\*(?::(\w+))?>", lambda m: f" {m.group(1) or ''} ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # camelCase -> camel Case
    return re.findall(r"[a-z]+|\d{3}", text.lower())


def expand(words: list[str]) -> list[str]:
    out = list(words)
    joined = "".join(words)
    for key, more in SYNONYMS.items():
        if key in words or (len(key) > 6 and key in joined):
            out += more.split()
    return out


@dataclass
class TemplateIndex:
    version: int
    ids: list[int]
    texts: list[str]
    kind: str  # "minilm" | "tfidf"
    text_vectors: np.ndarray
    idf: dict[str, float] = field(default_factory=dict)
    logbert: dict[int, np.ndarray] = field(default_factory=dict)
    _model: object = None

    @classmethod
    def build(cls, templates: dict[int, str], version: int, scorers: dict | None = None, snapshot=None) -> TemplateIndex:
        ids, texts = list(templates), list(templates.values())
        model = _minilm()
        if model is not None and ids:
            vecs = np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)
            idx = cls(version, ids, texts, "minilm", vecs, _model=model)
        else:
            docs = [expand(split_words(t)) for t in texts]
            df = Counter(w for d in docs for w in set(d))
            idf = {w: math.log((1 + len(docs)) / (1 + n)) + 1 for w, n in df.items()}
            idx = cls(version, ids, texts, "tfidf", np.zeros((0, 0), np.float32), idf)
            idx.text_vectors = np.stack([idx._tfidf(d) for d in docs]) if docs else np.zeros((0, 0), np.float32)
        idx.logbert = _logbert_vectors(scorers or {}, snapshot)
        return idx

    def _vocab(self) -> list[str]:
        if not hasattr(self, "_words"):
            self._words = sorted(self.idf)
            self._pos = {w: i for i, w in enumerate(self._words)}
        return self._words

    def _tfidf(self, words: list[str]) -> np.ndarray:
        self._vocab()
        v = np.zeros(len(self._words), np.float32)
        for w, n in Counter(words).items():
            if w in self._pos:
                v[self._pos[w]] = (1 + math.log(n)) * self.idf[w]
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v

    def embed(self, text: str) -> np.ndarray:
        if self.kind == "minilm":
            return np.asarray(self._model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)
        return self._tfidf(expand(split_words(text)))

    def search(self, text: str, k: int = 5) -> list[tuple[int, float]]:
        if not self.ids or not text.strip():
            return []
        sims = self.text_vectors @ self.embed(text)
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], round(float(sims[i]), 3)) for i in order if sims[i] > 0]

    def neighbours(self, tid: int, k: int = 5) -> list[tuple[int, float]]:
        """Templates the LogBERT models place near ``tid`` (seen in similar contexts)."""
        if tid not in self.logbert:
            return []
        me = self.logbert[tid]
        sims = [(t, float(v @ me)) for t, v in self.logbert.items() if t != tid]
        return [(t, round(s, 3)) for t, s in sorted(sims, key=lambda x: -x[1])[:k]]


def _minilm():
    """The sentence model when installed (``uv sync --extra ask``) and not switched off."""
    if os.environ.get("LOGPIPE_ASK_EMBEDDINGS", "auto") == "tfidf":
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(os.environ.get("LOGPIPE_ASK_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    return _MODEL


_MODEL = None


def _logbert_vectors(scorers: dict, snapshot) -> dict[int, np.ndarray]:
    sums: dict[int, list[np.ndarray]] = {}
    for scorer in scorers.values():
        table = scorer.model.embedding.token.weight.detach().cpu().numpy()
        for tid in scorer.vocab.template_ids:
            sums.setdefault(snapshot.canonical_id(tid) if snapshot else tid, []).append(table[scorer.vocab.index(tid)])
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
    if index is not None and index.kind == "minilm":  # TF-IDF scores are word overlap: the regex covers them
        for phrase in SHUTDOWN_PHRASES:
            for tid, score in index.search(phrase, 10):
                if score >= SHUTDOWN_MIN_SCORE and tid not in found:
                    found[tid] = f"close to {phrase!r} ({score:.2f})"
    return found


def shutdown_set(control, templates: dict[int, str], index: TemplateIndex | None) -> set[int]:
    edits = control.get("ask_shutdown", "edits") or {}
    auto = set(auto_shutdown(templates, index))
    return (auto | set(edits.get("add", []))) - set(edits.get("remove", []))
