"""Mask synthesis through a hosted chat model, or a signed-in chat web app.

``provider`` names one of :mod:`deepparse.providers` — ``anthropic``, ``openai``, ``gemini``
or ``groq``, keyed from the environment — or ``web``, which drives the user's own signed-in
browser through :mod:`deepparse.ai_scraper` and needs no key.

The model returns masks through the provider's structured output, so there is no free-form
parsing here. Each pattern is still compiled and de-duplicated before it becomes a
:class:`~deepparse.masks_types.Mask`, and the paper's safety net backfills the four core
classes, exactly as the Hugging Face backend does.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from deepparse.logging_utils import get_logger
from deepparse.masks_types import Mask
from deepparse.providers import ChatProvider, get_provider
from deepparse.synth.hf_deepseek_r1 import _ensure_core_classes
from deepparse.synth.prompt_templates import LLM_MASK_SYSTEM_PROMPT, LLM_MASK_USER_PROMPT

LOGGER = get_logger(__name__)

# Where each registered provider's key is read from. The first variable that is set wins.
API_KEY_ENV: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "groq": ("GROQ_API_KEY",),
}

WEB_PROVIDER = "web"


class MissingApiKeyError(RuntimeError):
    """The chosen provider needs a key and none of its environment variables is set."""


class _MaskOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(description="Short upper-case class name, e.g. TIMESTAMP, IPV4, BLK.")
    pattern: str = Field(description="A Python `re` pattern matching only the dynamic part.")
    justification: str = Field(description="One sentence on what the pattern captures.")


class MaskResponse(BaseModel):
    """The structured reply the model is held to."""

    model_config = ConfigDict(extra="forbid")

    masks: list[_MaskOut]


@dataclass
class LLMSynthConfig:
    provider: str = "anthropic"
    model_name: str | None = None
    effort: str = "medium"
    max_tokens: int = 8192
    timeout: float = 300.0
    max_retries: int = 2
    seed: int | None = None
    self_consistency_attempts: int = 2
    # Only read for provider="web".
    browser: str = "camoufox"
    headless: bool = True


def api_key_for(provider: str) -> str:
    names = API_KEY_ENV.get(provider, ())
    for name in names:
        if key := os.environ.get(name):
            return key
    raise MissingApiKeyError(
        f"Provider '{provider}' needs an API key; set {' or '.join(names) or 'one'}."
    )


def _load_provider(config: LLMSynthConfig) -> tuple[ChatProvider, str]:
    name = config.provider.strip().lower()
    if name == WEB_PROVIDER:
        # Imported here: the browser stack (playwright, camoufox) is an optional extra.
        from deepparse.providers.web import WebProvider

        web = WebProvider(browser=config.browser, headless=config.headless, log=LOGGER.info)
        return web, ""
    return get_provider(name), api_key_for(name)


def _to_masks(reply: MaskResponse) -> list[Mask]:
    masks: list[Mask] = []
    seen: set[str] = set()
    for item in reply.masks:
        if not item.pattern or item.pattern in seen:
            continue
        seen.add(item.pattern)
        try:
            re.compile(item.pattern)
        except re.error:
            LOGGER.warning("dropping invalid pattern: %r", item.pattern)
            continue
        masks.append(
            Mask(
                label=item.label.strip().upper() or f"VAR{len(masks)}",
                pattern=item.pattern,
                justification=item.justification or "LLM-synthesised",
            )
        )
    return masks


def _log_usage(provider: ChatProvider, model: str, raw: object) -> None:
    usage = getattr(raw, "usage_metadata", None) or {}
    tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    try:
        price = provider.price(model)
    except Exception as error:  # a pricing lookup must never fail a synthesis
        LOGGER.debug("price lookup for %s failed: %s", model, error)
        price = None
    cost = f"${price.cost(tokens_in, tokens_out):.4f}" if price else "unknown cost"
    LOGGER.info("%s/%s used %d in / %d out tokens (%s)",
                provider.name, model, tokens_in, tokens_out, cost)


def synthesize_llm(logs: Sequence[str], config: LLMSynthConfig | None = None) -> list[Mask]:
    """Ask a chat provider for masks covering ``logs`` (already sampled)."""
    config = config or LLMSynthConfig()
    provider, api_key = _load_provider(config)
    model_name = config.model_name or provider.default_model
    LOGGER.info("Synthesising masks with %s/%s", provider.name, model_name)

    chat = provider.build(
        api_key=api_key,
        model=model_name,
        max_tokens=config.max_tokens,
        effort=config.effort,
        timeout=config.timeout,
        max_retries=config.max_retries,
        seed=config.seed,
    )
    runnable = provider.structured(chat, MaskResponse)
    messages = [
        SystemMessage(LLM_MASK_SYSTEM_PROMPT),
        HumanMessage(LLM_MASK_USER_PROMPT.format(logs="\n".join(logs))),
    ]

    masks: list[Mask] = []
    try:
        # Paper's self-consistency check: re-prompt while the reply yields no usable mask.
        for attempt in range(1, config.self_consistency_attempts + 1):
            result = runnable.invoke(messages)
            _log_usage(provider, model_name, result.get("raw"))
            parsed = result.get("parsed")
            if parsed is None:
                LOGGER.warning("attempt %d: reply did not fit the schema: %s",
                               attempt, result.get("parsing_error"))
                continue
            masks = _to_masks(parsed)
            if masks:
                break
            LOGGER.warning("attempt %d: no valid masks in the reply", attempt)
    finally:
        if close := getattr(provider, "close", None):
            close()

    return _ensure_core_classes(masks)
