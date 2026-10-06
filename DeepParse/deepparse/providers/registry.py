"""Finding a provider by name.

A dictionary, and that is the point. The API stores a provider name against each author's
key; this turns that name into something the pipeline can call. Neither end holds a
reference to a vendor module.
"""

from __future__ import annotations

from functools import lru_cache

from deepparse.providers.anthropic import AnthropicProvider
from deepparse.providers.base import ChatProvider
from deepparse.providers.gemini import GeminiProvider
from deepparse.providers.groq import GroqProvider
from deepparse.providers.openai import OpenAIProvider


class UnknownProviderError(LookupError):
    """The stored credential names a provider this build does not carry."""


@lru_cache(maxsize=1)
def _registry() -> dict[str, ChatProvider]:
    providers: list[ChatProvider] = [
        AnthropicProvider(),
        OpenAIProvider(),
        GeminiProvider(),
        GroqProvider(),
    ]
    return {provider.name: provider for provider in providers}


def get_provider(name: str) -> ChatProvider:
    registry = _registry()
    key = name.strip().lower()

    if key not in registry:
        raise UnknownProviderError(
            f"No provider named '{name}' is registered. This build carries: "
            f"{', '.join(sorted(registry))}."
        )

    return registry[key]


def available() -> list[str]:
    return sorted(_registry())
