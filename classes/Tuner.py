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

    def readyforupdate(self, generation: int, lookback: int) -> bool:
        return generation - self.last_update >= lookback

    def updateHyperparameter(self, key: str, value: float, generation: int, lookback):
        self.hyperparamters.update({key: value})
        self.last_update = generation

        return self.hyperparamters

    def hasProgressed(self, name: str, metrics: list, lookback: int, threshold: float) -> bool:
        vals = metrics[-lookback:]
        return np.max(vals) - np.min(vals) > threshold

    def diversity_low(self, weights, threshold: float) -> bool:
        diversity = 0
        for i in range(weights.shape[1]):
            diversity += np.std(weights[:, i])
        return diversity < threshold

    def noMeanMaxDifference(self, mean_fitness, max_fitness, threshold):
        """
        The difference between the max fitness and mean fitness is very close together.
            PROBLEM: All individuals in the population look like each other
        """
        pass

    def noMaxIncrease(self, mean_fitness, max_fitness, threshold):
        """
        The difference between the max fitness and mean fitness is very close together.
            PROBLEM: All individuals in the population look like each other
        """
        pass
