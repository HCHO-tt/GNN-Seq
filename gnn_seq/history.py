"""Causal history attached to persistent decomposition subproblems."""
from collections import deque
import numpy as np


class SubproblemHistory:
    def __init__(self, size):
        self.next_uid = size
        self.uids = list(range(size))
        self.records = {uid: self.empty() for uid in self.uids}
        self.begin()

    @staticmethod
    def empty():
        return {"u": deque(maxlen=20), "s": deque(maxlen=20),
                "c": deque(maxlen=20), "op": deque(maxlen=20),
                "stall": 0, "age": 0}

    def begin(self):
        self.incoming = {}
        self.outgoing = {}
        self.generated = {}
        self.successful = {}
        self.replaced = set()
        self.events = []

    def child(self, source, replacements):
        """One child can replace multiple targets; success is counted once."""
        self.generated[source] = self.generated.get(source, 0) + 1
        self.successful[source] = self.successful.get(source, 0) + bool(replacements)
        for target, old, new in replacements:
            gain = float(np.clip((old - new) / (abs(old) + 1e-12), 0, 1))
            self.incoming[target] = self.incoming.get(target, 0.0) + gain
            self.outgoing[source] = self.outgoing.get(source, 0.0) + gain
            self.replaced.add(target)
            self.events.append({"source_uid": source, "target_uid": target,
                                "old_fitness": float(old), "new_fitness": float(new),
                                "relative_gain": gain})

    def remap(self, old_indices):
        """None denotes a newly created weight, even if its numbers coincide."""
        previous = self.uids
        self.uids = []
        for index in old_indices:
            if index is None:
                uid = self.next_uid
                self.next_uid += 1
                self.records[uid] = self.empty()
            else:
                uid = previous[index]
            self.uids.append(uid)
        self.records = {uid: self.records[uid] for uid in self.uids}

    def finish(self, actions=None):
        for uid in self.uids:
            # Newly introduced weights did not participate in this generation.
            if uid not in self.generated:
                continue
            item = self.records[uid]
            item["u"].append(float(-np.expm1(-self.incoming.get(uid, 0.0))))
            item["c"].append(float(-np.expm1(-self.outgoing.get(uid, 0.0))))
            item["s"].append(self.successful.get(uid, 0) / self.generated[uid])
            if actions is not None:
                item["op"].append((tuple(int(a) for a in actions), item["c"][-1]))
            item["stall"] = 0 if uid in self.replaced else item["stall"] + 1
            item["age"] += 1

    def operator_features(self):
        """Past action frequency and observed source gain, never current outcome."""
        rows = []
        for uid in self.uids:
            records = self.records[uid]["op"]
            values = np.zeros((4, 4, 2), np.float32)
            for agent in range(4):
                for action in range(4):
                    selected = [gain for joint, gain in records if joint[agent] == action]
                    values[agent, action] = (
                        len(selected) / 20,
                        float(np.mean(selected)) if selected else 0.)
            rows.append(values.reshape(-1))
        return np.asarray(rows, dtype=np.float32)

    def features(self, generations):
        def mean(values, window):
            return float(np.mean(list(values)[-window:])) if values else 0.0
        return np.asarray([
            [mean(r["u"], 5), mean(r["u"], 20),
             r["stall"] / max(generations, 1), mean(r["s"], 5),
             mean(r["c"], 5), min(r["age"], 20) / 20]
            for r in (self.records[uid] for uid in self.uids)
        ], dtype=np.float32)
