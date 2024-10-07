from typing import Dict
import numpy as np

"""
    Opties:
        willen we per fitness-regel een variable 'lookback' of houden we een algemene bij?

"""


class Tuner:
    def __init__(self, hyperparameters) -> None:
        self.hyperparamters = hyperparameters
        self.tracker: Dict = {}
        self.last_update = 0

    def updateHyperparameter(self, key: str, value: float, generation: int, lookback):

        if generation - self.last_update > lookback:
            self.hyperparamters.update({key: value})
            self.last_update = generation

        return self.hyperparamters

    def hasProgressed(self, name: str, metrics: list, lookback: int, threshold: float) -> bool:

        if not self.tracker.get(name):
            self.tracker[name] = {"name": name, "last_update": 0}

        vals = metrics[-lookback:]

        print(vals, np.max(vals) - np.min(vals))

        return np.max(vals) - np.min(vals) > threshold
