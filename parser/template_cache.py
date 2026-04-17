from ingestion.format_detector import Record, Ingestor

class TemplateCache:
    def __init__(self):
        self.cache = {}

    def _key(self, tokens):
        # first token + length
        return (tokens[0], len(tokens)) if tokens else None

    def get(self, tokens):
        return self.cache.get(self._key(tokens))

    def put(self, tokens, group):
        self.cache[self._key(tokens)] = group