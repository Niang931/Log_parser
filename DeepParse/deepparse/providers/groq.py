from __future__ import annotations

from decimal import Decimal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from pydantic import SecretStr

from deepparse.providers.base import Effort, ModelPrice

# TODO: add an automatic extractor later, for now the structure is hard to extract so
# hard-coded for now
# USD per million tokens, from https://console.groq.com/docs/models (September 2026). The
# Llama models are listed there without a price, so they cost None — see ChatProvider.price.
_PRICES: dict[str, ModelPrice] = {
    "openai/gpt-oss-120b": ModelPrice(Decimal("0.15"), Decimal("0.60")),
    "openai/gpt-oss-20b": ModelPrice(Decimal("0.075"), Decimal("0.30")),
    "qwen/qwen3.8-27b": ModelPrice(Decimal("0.80"), Decimal("4.00")),
    "qwen/qwen3.6-27b": ModelPrice(Decimal("0.60"), Decimal("3.00")),
}

# Groq's reasoning models take low/medium/high; the two levels above fold into high.
_EFFORTS = {"max": "high", "xhigh": "high", "high": "high", "medium": "medium", "low": "low"}

# NOTE: only these accept reasoning_effort with low/medium/high. Sending it to a Llama model
# is a 400, and Qwen 3.6 only takes none/default.
_REASONING_PREFIXES = ("openai/gpt-oss-", "qwen/qwen3.8-")

# Groq enforces json_schema on these; everything else has to fall back to a tool call.
_JSON_SCHEMA_PREFIXES = ("openai/gpt-oss-", "qwen/qwen3.8-")


class GroqProvider:
    name = "groq"

    @property
    def default_model(self) -> str:
        return "openai/gpt-oss-120b"

    def price(self, model: str) -> ModelPrice | None:
        return _PRICES.get(model)

    def build(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int,
        effort: Effort,
        timeout: float,
        max_retries: int,
        seed: int | None = None,
    ) -> BaseChatModel:
        return ChatGroq(
            model=model,
            api_key=SecretStr(api_key),
            max_tokens=max_tokens,
            reasoning_effort=(
                _EFFORTS.get(effort, "medium") if model.startswith(_REASONING_PREFIXES) else None
            ),
            timeout=timeout,
            max_retries=max_retries,
            # NOTE: ChatGroq has no seed field; the API takes one, so it rides in model_kwargs
            model_kwargs={} if seed is None else {"seed": seed},
        )

    def structured(self, model: BaseChatModel, schema: type) -> Runnable:
        name = getattr(model, "model_name", "")

        # NOTE: strict json_schema where Groq supports it; a forced tool call elsewhere,
        # which every Groq chat model can do but only validates client-side
        if name.startswith(_JSON_SCHEMA_PREFIXES):
            return model.with_structured_output(
                schema, method="json_schema", strict=True, include_raw=True
            )

        return model.with_structured_output(schema, method="function_calling", include_raw=True)
