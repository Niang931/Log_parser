"""Template normalisation rules used by the corrected LogHub ground truth.

Khan et al., "Guidelines for Assessing the Accuracy of Log Message Template
Identification Techniques" (ICSE 2022) define the rules the corrected
Loghub-2k / Loghub-2.0 templates follow. Applying them to both sides before
comparing means PA measures variable identification, not formatting.

DS double spaces, PS path-like strings, BL booleans, US default user strings,
DG digit-only tokens, WV word concatenated with a variable, DV dot-separated
variables, CV consecutive variables.
"""
from __future__ import annotations

import re

WILDCARD = "<*>"
_USER_STRINGS = {"true", "false", "null", "root", "admin"}
# Path-like strings are found with a reduced delimiter set ("." and "-" stay inside paths).
_PATH_DELIMS = re.compile(r"""(\s|,|!|;|:|=|\||"|'|\[|\]|\(|\)|\{|\})""")
_TOKEN_DELIMS = re.compile(r"""(\s|,|!|;|:|=|\||"|'|\[|\]|\(|\)|\{|\}|\.|-|\+|@|#|\$|%|&)""")
_PATH = re.compile(r"^(/[^/]+)+$")
_WORD_WITH_VAR = re.compile(r"^[^\s/]*<\*>[^\s/]*$")


def normalise_template(template: str) -> str:
    template = re.sub(r"\s+", " ", template.strip())  # DS
    template = "".join(
        WILDCARD if _PATH.match(part) else part for part in _PATH_DELIMS.split(template)
    )  # PS
    parts = []
    for part in _TOKEN_DELIMS.split(template):
        if part.lower() in _USER_STRINGS or part.isdigit():  # BL, US, DG
            part = WILDCARD
        elif part != "<*>/<*>" and _WORD_WITH_VAR.match(part):  # WV
            part = WILDCARD
        parts.append(part)
    template = "".join(parts)
    for pattern in (r"<\*>\.<\*>", r"<\*><\*>"):  # DV, then CV last
        while (merged := re.sub(pattern, WILDCARD, template)) != template:
            template = merged
    return template
