from __future__ import annotations

from decimal import Decimal
from typing import Literal

import requests
from bs4 import BeautifulSoup
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import SecretStr

from deepparse.providers.base import ModelPrice

Effort = Literal["max", "xhigh", "high", "medium", "low"]


PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing"


def extract_price_number(text: str) -> float:
    """Extract the 10 from $ 10 / MTok"""
    return float(text.split("$", 1)[-1].split("/")[0].strip())


def extract_pricing(html: str) -> dict[str, dict]:
    """
    Extract the Claude model pricing table from HTML.

    Returns:
        {
        "claude-fable-5.1":
            {
                "usd_mtok_input": 10.0,
                "usd_mtok_output": 50.0,
            },
        ...
        }
    """
    soup = BeautifulSoup(html, "html.parser")

    # Find the table based on its column headers rather than
    # depending on CSS classes, which may change.
    table = None

    for candidate in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True) for th in candidate.find_all("th")]

        if "Model" in headers and "Base input tokens" in headers and "Output tokens" in headers:
            table = candidate
            break

    if table is None:
        raise ValueError("Claude pricing table not found")

    headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]

    model_to_price = {}

    table = table.find("tbody")
    assert table is not None

    for tr in table.find_all("tr"):
        cells = tr.find_all("td")

        if len(cells) != len(headers):
            continue

        values = [cell.get_text(" ", strip=True) for cell in cells]

        row = dict(zip(headers, values, strict=False))

        # Convert to langchain model name convention
        model_name = row["Model"].split("(", 1)[0].strip()
        model_name = "-".join(model_name.lower().split())

        model_to_price[model_name] = {
            "usd_mtok_input": extract_price_number(row["Base input tokens"]),
            "usd_mtok_output": extract_price_number(row["Output tokens"]),
        }

    return model_to_price


class AnthropicProvider:
    """The one provider this build ships wired up."""

    name = "anthropic"

    @property
    def default_model(self) -> str:
        return "claude-opus-5"

    def price(self, name: str) -> ModelPrice | None:
        response = requests.get(
            PRICING_URL,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        model_to_price = extract_pricing(response.text)
        price = model_to_price.get(name)
        if price:
            return ModelPrice(
                input_per_mtok=Decimal(price["usd_mtok_input"]),
                output_per_mtok=Decimal(price["usd_mtok_output"]),
            )

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
        # NOTE: no seed option for this one
        return ChatAnthropic(
            model_name=model,
            api_key=SecretStr(api_key),
            max_tokens_to_sample=max_tokens,
            thinking={"type": "adaptive"},
            effort=effort,
            timeout=timeout,
            max_retries=max_retries,
            stream_usage=True,
            stop=None,
        )

    def structured(self, model: BaseChatModel, schema: type) -> Runnable:
        # NOTE: method="json_schema" is an Anthropic specific argument, include raw
        # also specified to get the usage metadata
        return model.with_structured_output(schema, method="json_schema", include_raw=True)
