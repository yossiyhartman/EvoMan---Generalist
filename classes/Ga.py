import numpy as np


class Ga:

    def __init__(self) -> None:
        pass

    @classmethod
    def norm(self, x, pfit_pop):
        if (max(pfit_pop) - min(pfit_pop)) > 0:
            x_norm = (x - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
        else:
            x_norm = 0

        if x_norm <= 0:
            x_norm = 0.0000000001
        return x_norm

    @classmethod
    def normalize(self, x: np.array):
        return np.asarray([np.divide(k - np.mean(x), np.std(x)) for k in x])

    @classmethod
    def minmax(self, x: np.array):
        mi, ma = np.min(x), np.max(x)
        return np.asarray([np.divide(k - mi, ma - mi) for k in x])

    @classmethod
    def bound(self, x, lb: int = 2, ub: int = 2):
        if x > ub:
            x = x
        elif x < lb:
            x = -2
        return x

    ###################
    # INITIALIZING
    ###################

    def initialize_population(self, population_size: int, n_genomes: int, normal: bool = False) -> np.array:
        if normal:
            return np.random.normal(size=(population_size, n_genomes))
        else:
            return np.random.uniform(low=-2, high=2, size=(population_size, n_genomes))

    ###################
    # REPOPULATION
    ###################

    def repopulate(self, population: np.array, fitness: np.array, frac: float = 0.75):

        _, n_genomes = population.shape
        order = np.argsort(fitness)

        discard_amount = int(np.ceil(population.shape[0] * frac))

        new_pop = self.initialize_population(discard_amount, n_genomes)

        return np.vstack((population[order][discard_amount:], new_pop))

    ###################
    # SELECTION
    ###################

    def tournament_selection(self, population: np.array, fitness: np.array, tournament_size: int = 2) -> np.array:
        selected_w, selected_f = [], []

        population_size = population.shape[0]

        for i in range(0, population_size, 1):

            idx = np.random.randint(0, population_size, tournament_size)

            fitness_vals = fitness[idx]

            best_sample_idx = np.argmax(fitness_vals)

            winner_w = population[idx[best_sample_idx]]
            winner_f = fitness[idx[best_sample_idx]]

            selected_w.append(winner_w)
            selected_f.append(winner_f)

        return np.asarray(selected_w), np.asarray(selected_f)

    def eletist_selection(self, population: np.array, fitness: np.array, top: int = 2) -> np.array:

        idx_n_best_individuals = np.argpartition(fitness, -top)[-top:]
        n_best_individuals_w = population[idx_n_best_individuals]
        n_best_individuals_f = fitness[idx_n_best_individuals]

        population = np.delete(population, idx_n_best_individuals, axis=0)
        fitness = np.delete(fitness, idx_n_best_individuals, axis=0)

        return n_best_individuals_w, n_best_individuals_f, population, fitness

    def survival_selection(self, population: np.array, fitness: np.array, size: int) -> np.array:

        normalized_f = np.asarray(list(map(lambda x: self.norm(x, fitness), fitness)))

        # Calculate a survival probability
        survival_prob = normalized_f / np.sum(normalized_f)

        selection_idx = np.random.choice(population.shape[0], size=size, p=survival_prob, replace=False)

        return population[selection_idx], fitness[selection_idx]

    def roulette_wheel_selection(self, population: np.array, fitness: np.array) -> np.array:
        min_fitness = np.min(fitness)
        if min_fitness < 0:
            fitness = fitness - min_fitness

        total_fitness = np.sum(fitness)

        if total_fitness == 0:
            probabilities = np.ones_like(fitness) / len(fitness)
        else:
            probabilities = fitness / total_fitness

        indices = np.random.choice(len(population), size=self.population_size, p=probabilities)
        return population[indices]

    def rank_selection(self, population: np.array, fitness: np.array) -> np.array:
        ranks = np.argsort(fitness)
        rank_probabilities = np.arange(1, self.population_size + 1) / np.sum(np.arange(1, self.population_size + 1))
        selected_indices = np.random.choice(ranks, size=self.population_size, p=rank_probabilities)
        return population[selected_indices]

    def stochastic_universal_sampling(self, population: np.array, fitness: np.array) -> np.array:
        total_fitness = np.sum(fitness)

        if total_fitness == 0:
            return population

        distance = total_fitness / self.population_size
        start_point = np.random.uniform(0, distance)
        points = [start_point + i * distance for i in range(self.population_size)]

        cumulative_fitness = np.cumsum(fitness)
        selected = []
        i, j = 0, 0

        while i < self.population_size and j < len(population):
            if points[i] < cumulative_fitness[j]:
                selected.append(population[j])
                i += 1
            else:
                j += 1

        return np.asarray(selected)

    ###################
    # MUTATION
    ###################

    def mutate(self, offspring: np.array, p_mutation: float = 0.5, p_genome: float = 0.5, sigma_mutation: float = 0.3) -> np.array:

        for individual in offspring:
            if np.random.rand() < p_mutation:
                mutation_dist = np.random.uniform(0, 1, size=offspring.shape[1]) < p_genome
                individual += mutation_dist * np.random.normal(0, sigma_mutation, size=offspring.shape[1])
                individual = np.asarray(list(map(lambda x: self.bound(x), individual)))

        return offspring

    ###################
    # CROSSOVER
    ###################

    def crossover_2_offspring(self, parents: np.array, p: float = 0.5) -> np.array:

        total_offspring = []

        for i in range(0, parents.shape[0], 2):

            offspring = np.zeros(shape=(2, len(parents[i])))

            dist = np.random.uniform(0, 1, size=len(parents[i]))

            select_p1 = dist < p
            select_p2 = dist >= p

            offspring[0] = select_p1 * parents[i] + select_p2 * parents[i + 1]
            offspring[1] = select_p2 * parents[i] + select_p1 * parents[i + 1]

            total_offspring.append(offspring[0])
            total_offspring.append(offspring[1])

        return np.asarray(total_offspring)

    def crossover_n_offspring(self, parents: np.array, n_offspring: int = 4) -> np.array:

        total_offspring = []

        for i in range(0, parents.shape[0], 2):

            n_gnomes = len(parents[i])

            offspring = np.zeros(shape=(n_offspring, n_gnomes))

            for child in offspring:
                cross_distribution = np.random.uniform(0, 1, n_gnomes)
                child += cross_distribution * parents[i] + (1 - cross_distribution) * parents[i + 1]
                total_offspring.append(child)

        return np.asarray(total_offspring)
