"""Ask: a plain-English question becomes a validated :class:`~logpipe.ask.spec.QuerySpec`, the
read-only queries it needs are shown for approval, and the answer (a table, or a root-cause report
built from a small evidence bundle) cites the lines it rests on.

The LLM only routes the question and writes the final explanation; it never writes SQL and never
reads logs in bulk. Nothing here changes the parse/detect pipeline.
"""
