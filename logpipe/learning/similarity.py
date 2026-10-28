
from collections.abc import Sequence

from ..enrich.llm import LLMAdapter
from ..registry import TemplateEntry


def find_alias(
    candidate: tuple[str, ...],
    existing_templates: Sequence[TemplateEntry],
    adapter: LLMAdapter | None = None,
) -> int | None:
    """Return old template_id if it is alias, otherwise return None."""
    ...