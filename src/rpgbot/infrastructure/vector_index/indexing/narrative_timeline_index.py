from collections import defaultdict


class NarrativeTimelineIndex:

    def __init__(self):

        self.timeline = defaultdict(list)
        self.doc_time = {}


    def add(self, doc_id, timestamp):

        if timestamp is None:
            return

        self.timeline[timestamp].append(doc_id)
        self.doc_time[doc_id] = timestamp


    def neighbors(self, doc_id, window=1):

        ts = self.doc_time.get(doc_id)

        if ts is None:
            return []

        keys = sorted(self.timeline.keys())

        if ts not in keys:
            return []

        pos = keys.index(ts)

        start = max(0, pos - window)
        end = min(len(keys), pos + window + 1)

        result = []

        for t in keys[start:end]:
            result.extend(self.timeline[t])

        return result