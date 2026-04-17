import re
from format_detector import Ingestor, Record
from parser.parser import tokenize


class LogGroup:
    def __init__(self, log):
        self.logs = [log]
        self.template = tokenize(log)

    def add_log(self, log):
        self.logs.append(log)
        self.template =




class GroupingTree:

    def __init__(self):
        self.tree = {}

    def __str__(self):
        return f"{self.tree.items()}"

    def index_ts(self, log, ts_format = r"\d{2}:\d{2}:\d{2}"):
        match = re.search(ts_format, log)
        if match:
            timestamp_idx = match.end()
        else:
            timestamp_idx = 0
        return timestamp_idx


    def add_log(self, log):
        timestamp_idx = self.index_ts(log)
        new_log = log[timestamp_idx:]
        length = len(new_log)
        prefix = new_log[:3]

        if length not in self.tree.keys():
            self.tree[length]= {prefix:[log]}

        else:
            if prefix in self.tree[length].keys():
                self.tree[length][prefix].append(log)
            else:
                self.tree.get(length)[prefix] = [log]
        print(self.tree.keys())


tree = GroupingTree()
ingestor = Ingestor()
ingestor.create_record('log.txt')
print(ingestor)
tree.add_log('2026-04-13 10:15:23.123 [INFO] Thermostat SN:TH-45678 temp=22.5C target=23.0C mode=auto humidity=45%')
tree.add_log('2026-04-13 10:15:23.100 [INFO] Thermostat SN:TH-45678 temp=22.5C target=23.0C mode=auto humidity=45%')
tree.add_log('"template": "<> [<>] Thermostat SN:<> <>=<> <>=<> <>=<> <>=<*>",')
print(tree)