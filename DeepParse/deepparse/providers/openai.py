from __future__ import annotations

from decimal import Decimal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from deepparse.providers.base import Effort, ModelPrice

# TODO: add an automatic extractor later, for now the structure is hard to extract so
# hard-coded for now
# Standard tier, USD per million tokens, from https://developers.openai.com/api/docs/pricing
# (September 2026). Unlike Anthropic's pricing page, OpenAI's is rendered client-side and
# has nothing to scrape, so this is a table to update. A model missing from it is priced as
# None — see ChatProvider.price.
_PRICES: dict[str, ModelPrice] = {
    "gpt-6-astra": ModelPrice(Decimal("10.00"), Decimal("50.00")),
    "gpt-5.6-sol": ModelPrice(Decimal("4.00"), Decimal("20.00")),
    "gpt-5.6": ModelPrice(Decimal("4.00"), Decimal("20.00")),
    "gpt-5.6-terra": ModelPrice(Decimal("2.00"), Decimal("12.00")),
    "gpt-5.6-luna": ModelPrice(Decimal("0.20"), Decimal("1.20")),
    "gpt-5.5": ModelPrice(Decimal("5.00"), Decimal("30.00")),
    "gpt-5.4": ModelPrice(Decimal("2.50"), Decimal("15.00")),
    "gpt-5": ModelPrice(Decimal("1.25"), Decimal("10.00")),
}

# The pipeline speaks Anthropic's effort vocabulary. OpenAI's reasoning_effort tops out at
# xhigh, so max folds into it.
_EFFORTS = {"max": "xhigh", "xhigh": "xhigh", "high": "high", "medium": "medium", "low": "low"}


class OpenAIProvider:
    name = "openai"

    @property
    def default_model(self) -> str:
        return "gpt-6-astra"

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
        return ChatOpenAI(
            model=model,
            api_key=SecretStr(api_key),
            max_tokens=max_tokens,
            reasoning_effort=_EFFORTS.get(effort, "medium"),
            timeout=timeout,
            max_retries=max_retries,
            seed=seed,
            stream_usage=True,
        )

    def structured(self, model: BaseChatModel, schema: type) -> Runnable:
        # NOTE: json_schema is OpenAI's native structured output, enforced server-side
        return model.with_structured_output(
            schema, method="json_schema", strict=True, include_raw=True
        )
