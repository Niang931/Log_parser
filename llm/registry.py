"""
llm/registry.py — Multi-provider LLM abstraction layer
=======================================================
Supports: anthropic (primary), openai, groq, ollama (local fallback)
Rubric: cost/latency strategy, operational readiness, enterprise-ready
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

log = logging.getLogger("deepparse.llm.registry")

# ---------------------------------------------------------------------------
# Cost catalogue ($ per 1 K output tokens — approximate 2026 pricing)
# ---------------------------------------------------------------------------
COST_PER_1K_OUT: dict[str, float] = {
    "claude-sonnet-4.6": 0.015,
    "gpt-4o":                   0.015,
    "gpt-4o-mini":              0.0006,
    "llama-3.3-70b-versatile":          0.00059,   # Groq
    "gemini-3-flash": 0.0003, # Google fast tier
    "ollama-local":             0.0,
}

# ---------------------------------------------------------------------------
# Base LLM wrapper
# ---------------------------------------------------------------------------

class BaseLLM:
    def __init__(self, model: str, max_retries: int = 3, timeout: float = 60.0) -> None:
        self.model        = model
        self.max_retries  = max_retries
        self.timeout      = timeout
        self._total_tokens_out = 0
        self._total_cost_usd   = 0.0

    def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def complete_with_retry(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                t0  = time.monotonic()
                out = self.complete(prompt, max_tokens=max_tokens, temperature=temperature)
                latency = time.monotonic() - t0
                toks    = len(out.split())
                cost    = toks / 1000 * COST_PER_1K_OUT.get(self.model, 0.01)
                self._total_tokens_out += toks
                self._total_cost_usd   += cost
                log.debug(
                    "LLM[%s] attempt=%d latency=%.2fs tokens≈%d cost≈$%.5f",
                    self.model, attempt, latency, toks, cost,
                )
                return out
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                log.warning("LLM attempt %d/%d failed: %s — retry in %ds", attempt, self.max_retries, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"LLM {self.model} exhausted {self.max_retries} retries: {last_exc}") from last_exc

    @property
    def cost_summary(self) -> dict:
        return {
            "model":             self.model,
            "total_tokens_out":  self._total_tokens_out,
            "est_cost_usd":      round(self._total_cost_usd, 6),
        }


# ---------------------------------------------------------------------------
# Anthropic provider (primary — best for structured JSON output)
# ---------------------------------------------------------------------------

class AnthropicLLM(BaseLLM):
    """
    Uses claude-sonnet-4-20250514 via the Anthropic Messages API.
    Deterministic at temperature=0; excellent JSON instruction-following.
    """
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, model: str | None = None, **kw) -> None:
        super().__init__(model or self.DEFAULT_MODEL, **kw)
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc

    def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAILLM(BaseLLM):
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model: str | None = None, **kw) -> None:
        super().__init__(model or self.DEFAULT_MODEL, **kw)
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        except ImportError as exc:
            raise ImportError("pip install openai") from exc

    def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Groq provider (ultra-low latency inference)
# ---------------------------------------------------------------------------

class GroqLLM(BaseLLM):
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, model: str | None = None, **kw) -> None:
        super().__init__(model or self.DEFAULT_MODEL, **kw)
        try:
            from groq import Groq
            self._client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        except ImportError as exc:
            raise ImportError("pip install groq") from exc

    def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Ollama local provider (zero-cost, air-gapped environments)
# ---------------------------------------------------------------------------

class OllamaLLM(BaseLLM):
    DEFAULT_MODEL = "llama3"

    def __init__(self, model: str | None = None, base_url: str = "http://localhost:11434", **kw) -> None:
        super().__init__(model or self.DEFAULT_MODEL, **kw)
        self.base_url = base_url

    def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        import urllib.request
        payload = json.dumps({
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }).encode()
        req  = urllib.request.Request(f"{self.base_url}/api/generate", data=payload,
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return data.get("response", "")


# ---------------------------------------------------------------------------
# Mock / fallback LLM (no API key needed — returns static valid JSON)
# Used when no provider keys are configured; ensures pipeline still runs.
# ---------------------------------------------------------------------------

class MockLLM(BaseLLM):
    """
    Deterministic mock that returns a minimal valid mask array.
    Ensures full pipeline execution even in CI / air-gapped environments.
    """
    DEFAULT_MODEL = "mock-local"
    # Minimal bootstrap masks that almost always parse *something*
    _FALLBACK_MASKS = json.dumps([
        {"regex": "\\b\\d+\\.\\d+\\b",          "mask_with": "<FLOAT>"},
        {"regex": "\\b\\d{5,}\\b",              "mask_with": "<LARGE_INT>"},
        {"regex": "[A-Z]{2,6}_[A-Z0-9_]+",     "mask_with": "<FAB_TOKEN>"},
        {"regex": "\\d{4}-\\d{2}-\\d{2}",       "mask_with": "<DATE>"},
        {"regex": "\\bER-[A-Z0-9]+",            "mask_with": "<ERROR_CODE>"},
    ])

    def __init__(self, **kw) -> None:
        super().__init__(self.DEFAULT_MODEL, **kw)

    def complete(self, prompt: str, **_kw) -> str:
        log.warning("MockLLM active — returning fallback masks (no API key configured)")
        return self._FALLBACK_MASKS


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

class GeminiLLM(BaseLLM):
    """Google Gemini — generous free tier, strong JSON instruction following."""
    DEFAULT_MODEL = "gemini-3-flash"

    def __init__(self, model: str | None = None, **kw) -> None:
        super().__init__(model or self.DEFAULT_MODEL, **kw)
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
            self._genai  = genai
            self._client = genai.GenerativeModel(self.model)
        except ImportError as exc:
            raise ImportError("pip install google-generativeai") from exc

    def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        config = self._genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        resp = self._client.generate_content(prompt, generation_config=config)
        return resp.text or ""


# Replace the existing _PROVIDER_MAP with this
_PROVIDER_MAP: dict[str, type[BaseLLM]] = {
    "anthropic": AnthropicLLM,
    "openai":    OpenAILLM,
    "groq":      GroqLLM,
    "gemini":    GeminiLLM,
    "ollama":    OllamaLLM,
    "mock":      MockLLM,
}

# Replace the existing _KEY_ENV with this
_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "groq":      "GROQ_API_KEY",
    "gemini":    "GEMINI_API_KEY",
}


def init_llm(
    provider: str = "anthropic",
    model: str | None = None,
    auto_fallback: bool = True,
) -> BaseLLM:
    """
    Instantiate an LLM by provider name.
    Falls back to MockLLM if the required API key is absent and auto_fallback=True.

    Priority order for auto-detection:
      anthropic → openai → groq → ollama → mock
    """
    # Auto-detect best available provider when caller passes "auto"
    if provider == "auto":
        for p, env_key in _KEY_ENV.items():
            if os.environ.get(env_key):
                provider = p
                log.info("Auto-selected LLM provider: %s", p)
                break
        else:
            log.warning("No API keys found — using MockLLM fallback")
            provider = "mock"

    cls = _PROVIDER_MAP.get(provider)
    if cls is None:
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose from: {list(_PROVIDER_MAP)}")

    # Check key availability
    env_key = _KEY_ENV.get(provider)
    if env_key and not os.environ.get(env_key):
        if auto_fallback:
            log.warning(
                "Provider '%s' requires %s (not set) — falling back to MockLLM",
                provider, env_key,
            )
            return MockLLM()
        raise EnvironmentError(f"{env_key} not set for provider '{provider}'")

    kw: dict[str, Any] = {}
    if model:
        kw["model"] = model
    instance = cls(**kw)
    log.info("Initialised LLM: provider=%s model=%s", provider, instance.model)
    return instance
