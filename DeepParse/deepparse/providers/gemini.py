from __future__ import annotations

from decimal import Decimal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from deepparse.providers.base import Effort, ModelPrice

# TODO: add an automatic extractor later, for now the structure is hard to extract so
# hard-coded for now
# Paid tier, prompts up to 200k tokens, USD per million tokens, from
# https://ai.google.dev/gemini-api/docs/pricing (September 2026). A table to update, like
# OpenAI's. Longer prompts are billed higher by Google, which this under-counts — the safe
# direction for a budget ceiling. The 3.6-3.8 Flash prices are promotional until 2027-01-01,
# when they rise to $1.50 / $7.50.
_PRICES: dict[str, ModelPrice] = {
    "gemini-3.8-flash": ModelPrice(Decimal("0.75"), Decimal("3.75")),
    "gemini-3.7-flash": ModelPrice(Decimal("0.75"), Decimal("3.75")),
    "gemini-3.6-flash": ModelPrice(Decimal("0.75"), Decimal("3.75")),
    "gemini-3.5-flash": ModelPrice(Decimal("1.50"), Decimal("9.00")),
    "gemini-3.5-flash-lite": ModelPrice(Decimal("0.30"), Decimal("2.50")),
    "gemini-3.1-flash-lite": ModelPrice(Decimal("0.25"), Decimal("1.50")),
    "gemini-3.1-pro-preview": ModelPrice(Decimal("2.00"), Decimal("12.00")),
    "gemini-2.5-pro": ModelPrice(Decimal("1.25"), Decimal("10.00")),
    "gemini-2.5-flash": ModelPrice(Decimal("0.30"), Decimal("2.50")),
    "gemini-2.5-flash-lite": ModelPrice(Decimal("0.10"), Decimal("0.40")),
}

# Gemini's reasoning_effort stops at high, so the two levels above it fold into it.
_EFFORTS = {"max": "high", "xhigh": "high", "high": "high", "medium": "medium", "low": "low"}


class GeminiProvider:
    name = "gemini"

    @property
    def default_model(self) -> str:
        # NOTE: the newest stable model rather than 3.1 Pro, which is still a preview and
        # could be withdrawn from under a stored credential.
        return "gemini-3.8-flash"

    def price(self, model: str) -> ModelPrice | None:
        # The Models API reports ids as "models/<id>"; accept either spelling.
        return _PRICES.get(model.removeprefix("models/"))

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
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=SecretStr(api_key),
            max_output_tokens=max_tokens,
            reasoning_effort=_EFFORTS.get(effort, "medium"),
            timeout=timeout,
            max_retries=max_retries,
            seed=seed,
        )

    def structured(self, model: BaseChatModel, schema: type) -> Runnable:
        # NOTE: json_schema uses Gemini's response_schema, so the constraint is applied by
        # the model rather than parsed and hoped for afterwards
        return model.with_structured_output(schema, method="json_schema", include_raw=True)
