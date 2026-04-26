import re
import uuid
import os
from template_cache import TemplateCache
from ingestion.format_detector import Ingestor, Record

def tokenize(log):
    return log.strip().split()

def jaccard_similarity(token1, token2):
    return len(set(token1) & set(token2)) / len(set(token1) | set(token2))

def merge_template(template_tokens, new_tokens):
    merged = []
    for t1, t2 in zip(template_tokens, new_tokens):
        if t1 == t2:
            merged.append(t1)
        else:
            merged.append('<*>')
    return merged

def similarity(template_tokens, log_tokens):
    matched = 0
    for t1, t2 in zip(template_tokens, log_tokens):
        if t1 == t2 or t1 == '<*>':
            matched += 1
    return matched / len(template_tokens)


def find_template(log1, log2):
    token1 = tokenize(log1)
    token2 = tokenize(log2)
    if jaccard_similarity(token1, token2) > 0.1:
        template = merge_template(token1, token2)
    else:
        template = []
    print(jaccard_similarity(token1, token2))
    print(template)

class LogGroup:
    def __init__(self, group_id, tokens):
        self.group_id = group_id
        self.template = tokens[:]
        self.size = 1

    def update(self, tokens):
        self.template = merge_template(self.template, tokens)
        self.size += 1

    # def add_log(self, log):
    #     tokens = tokenize(log)
    #     self.template = merge_template(self.template, tokens)
    #     self.logs.append(log)

class LogParser:
    def __init__(self, threshold = 0.6):
        self.groups = []
        self.cache = TemplateCache()
        self.threshold = threshold

    def process(self, record):
        tokens = tokenize(record.text)
        record.tokens = tokens

        group = self.cache.get(tokens)

        if not group:
            best_group = None
            best_score = 0

            for g in self.groups:
                score = similarity(g.template, tokens)
                if score > best_score:
                    best_score = score
                    best_group = g

            if best_score > self.threshold:
                group = best_group
            else:
                group = LogGroup(str(uuid.uuid4()), tokens)
                self.groups.append(group)
        group.update(tokens)
        self.cache.put(tokens, group)

        record.group_id = group.group_id
        record.template = group.template

        return record




ingestor = Ingestor()
parser = LogParser()


for record in ingestor.walk(os.environ['LOGS_PATH']):
    print(record)
    enriched = parser.process(record)

    print("log",enriched.text)
    print("Template", " ".join(enriched.template))
    print("Group:", enriched.group_id)
    print("-" * 50)
