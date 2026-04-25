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
