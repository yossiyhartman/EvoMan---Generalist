from typing import Dict
import numpy as np

"""
    Opties:
        willen we per fitness-regel een variable 'lookback' of houden we een algemene bij?

"""


class Tuner:
    def __init__(self) -> None:
        self.tracker: Dict = {}

    def hasProgressed(self, name: str, metrics: list, lookback: int, threshold: float) -> bool:

        if not self.tracker.get(name):
            self.tracker[name] = {"name": name, "last_update": 0}

        vals = metrics[-lookback:]

        if len(vals) < lookback:
            return

        return np.max(vals) - np.min(vals) > threshold
