import uuid, datetime
import os
from pathlib import Path

class Record:

    def __init__(self, record_id, source, timestamp, format_hint, text):
        self.record_id = record_id
        self.source = source
        self.timestamp = timestamp
        self.format_hint = format_hint
        self.text = text

        self.tokens = None
        self.template = None
        self.group_id = None

        self.parsed = None
        self.normalized = None
        self.events = None

    def __str__(self):
        return f"[{self.record_id}] {self.source} | {self.text[:50]}"


class Ingestor:

    def walk(self, root):

        for dirpath, dirname, filename in os.walk(root):
            for fname in sorted(filename):
                ext = Path(fname).suffix.lower()

                fpath = os.path.join(dirpath, fname)
                yield from self.ingest_file(fpath)

    def ingest_file(self, path):
        with open(path, 'r') as f:
            for line in f:
                yield Record(
                    record_id=str(uuid.uuid4()),
                    source=path,
                    text=line.strip(),
                    timestamp=None,
                    format_hint=Path(path).suffix.lower()
                )

    # def ingest_file(self, path):
    #     ext = Path(path).suffix.lower()
    #     record_id = uuid.uuid4()
    #     now = datetime.datetime.now()
    #     with open (path, 'r') as f:
    #         text = f.read()
    #
    #     yield Record(record_id, path, now, ext, text)



ingestor = Ingestor()
ingestor.walk('log.txt')

