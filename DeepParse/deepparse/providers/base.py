"""What the pipeline needs from a chat provider, and nothing else.

The pipeline never imports a vendor. It asks the registry for a provider by name and uses
the three things below, which is the whole of the coupling:

* a LangChain chat model, and a way to bind a schema to it that returns validated
  objects — the *way* differs per vendor, which is exactly why it is behind the interface,
* a price for the model, so a job can be costed without the caller knowing the vendor's
  units,
* the exception types worth retrying, because "rate limited" is spelled differently by
  every vendor and the retry policy should not have to care.

Adding a provider is one module implementing ``ChatProvider`` and one line in the registry.
See ``providers/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, runtime_checkable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

# The effort levels the pipeline asks for, in the vocabulary of the models that have one.
# A provider without the concept maps these onto whatever it does have, or ignores them.
Effort = str


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """What a model costs, per million tokens."""

    input_per_mtok: Decimal
    output_per_mtok: Decimal

    def cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        million = Decimal(1_000_000)

        return (
            Decimal(input_tokens) * self.input_per_mtok / million
            + Decimal(output_tokens) * self.output_per_mtok / million
        )


@runtime_checkable
class ChatProvider(Protocol):
    """One vendor's chat models, as the pipeline sees them."""

    name: str

    @property
    def default_model(self) -> str: ...

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
        """A chat model ready to be called.

        ``seed`` asks for repeatable sampling where the vendor offers it. An adapter whose
        vendor has no seed ignores it rather than failing: reproducibility is best effort.
        """
        ...

    def structured(self, model: BaseChatModel, schema: type) -> Runnable:
        """Binds a schema so the call returns a validated object.

        Whether that is the vendor's native structured output, a forced tool call, or a
        JSON-mode parser is the adapter's business — the pipeline only ever sees an instance
        of ``schema`` come back, alongside the raw message it needs for token counts.

        Must return a runnable whose output is ``{"raw": AIMessage, "parsed": schema | None,
        "parsing_error": Exception | None}`` — LangChain's ``include_raw=True`` shape. The
        raw message is not optional: it is where the usage lives, and a pipeline that could
        not cost its own calls could not be budgeted.
        """
        ...

    def price(self, model: str) -> ModelPrice | None:
        """What the model costs, or None when this build does not know.

        None is a real answer, not a failure: a model released after this table was written
        should run and be reported as costing nothing known, rather than refusing to run.
        The budget then under-counts, which is the safe direction to be wrong in for a
        ceiling whose other half is a request count.
        """
        ...
