from typing import Dict
import numpy as np


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

    def noMaxIncrease(self, max_fitness, threshold: float = 10.0, lookback: int = 0):
        """
        The max hasn't inreased in 'n' generations
            PROBLEM: stuck in a local optimum
        """
        return np.subtract(max_fitness[-1], max_fitness[-lookback]) <= threshold

    def noMeanMaxDifference(self, mean_fitness, max_fitness, threshold: float = 10.0, lookback: int = 0):
        """
        The difference between the max fitness and mean fitness is very close together.
            PROBLEM: All individuals in the population look like each other
        """
        return all(np.subtract(max_fitness[-lookback:], mean_fitness[-lookback:]) <= threshold)

    def similairWeights(self, population):
        """
        The weights of all genomes look very similar
            PROBLEM: All individuals in the population look like each other
        """
        distances = np.zeros(shape=(population.shape[0], population.shape[0]))

        for i in range(population.shape[0]):
            for j in range(population.shape[0]):
                distances[i, j] = np.linalg.norm(np.subtract(population[i], population[j])) / 65

        return distances
