# Double-Blind Safety Guide

This repository was constructed to satisfy anonymous artifact evaluation. To verify compliance:

1. Search the tree for proper nouns or institution names (expect none).
2. Inspect `artifacts/outputs/` for hostnames or user paths before sharing.
3. Use `python anonymization/scrub_paths.py <input> <output>` to remove user-specific tokens from logs or CSV files.
4. Review shell scripts to ensure no network calls leak identity; dataset fetch scripts require manual download.
5. Confirm Git history does not include author metadata prior to submission.
