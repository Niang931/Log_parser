"""Prompt templates replicating the paper listings."""
from __future__ import annotations

MASK_SYNTH_PROMPT = """
You are DeepParse-RegSynth, a deterministic assistant that analyses server logs.
Given the following sample logs delimited by <LOGS>, produce a JSON array of objects.
Each object must contain fields: label, pattern (Python regex), and justification.
Only return valid JSON with double quoted keys.
<LOGS>
{logs}
</LOGS>
""".strip()

MASK_VALIDATION_PROMPT = """
You are verifying regex masks for log parsing. Ensure the JSON schema is respected
(label, pattern, justification). Reject overly generic patterns or those using `.*` unless
anchored to a prefix and suffix. Always output valid JSON.
""".strip()

# Chat-provider backend (deepparse.synth.chat_provider). The reply shape is enforced by the
# provider's structured output, so the prompt only has to say what makes a good mask.
LLM_MASK_SYSTEM_PROMPT = """
You are DeepParse-RegSynth. You write Python regular expressions ("masks") that capture
the dynamic, variable parts of log messages — timestamps, numbers, IP addresses, block
and request ids, paths, hex values, UUIDs — while leaving the static template text alone.

Rules:
- One mask per variable class; give each an upper-case label such as TIMESTAMP, LOGLEVEL,
  NUMBER, IPV4, HEX, UUID, PATH or a dataset-specific one like BLK.
- Patterns must compile with Python's `re` module and match only the variable token.
- Never use an unanchored `.*`; prefer explicit character classes and word boundaries.
- Do not write masks that would match ordinary English words in the template.
""".strip()

LLM_MASK_USER_PROMPT = """
Write masks for the dynamic fields in these sample log lines.
<LOGS>
{logs}
</LOGS>
""".strip()
