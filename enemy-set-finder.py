import json
import os
import datetime as dt
import numpy as np

from evoman.environment import Environment
from demo_controller import player_controller

# Our classes
from classes.Ga import Ga
from classes.Logger import Logger
from classes.Tuner import Tuner

n_hidden_neurons = 10
n_network_weights = (20 + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

os.environ["SDL_VIDEODRIVER"] = "dummy"

test_env = Environment(
    experiment_name="./",
    multiplemode="yes",
    enemies=[1, 2, 3, 4, 5, 6, 7, 8],
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),
    enemymode="static",
    level=2,
    speed="fastest",
    visuals=False,
)


def simulation(x, env):
    f, p, e, t = env.play(pcont=x)
    return f, p - e


def evaluate(x, env):
    evaluation = np.array(list(map(lambda y: simulation(y, env), x)))
    return evaluation[:, 0], evaluation[:, 1]


def calc_statistics(fitness: np.array, gain: np.array, prefix="train"):
    return {
        f"{prefix}_max.fitness": np.round(np.max(fitness), 6),
        f"{prefix}_mean.fitness": np.round(np.mean(fitness), 6),
        f"{prefix}_min.fitness": np.round(np.min(fitness), 6),
        f"{prefix}_std.fitness": np.round(np.std(fitness), 6),
        f"{prefix}_max.gain": np.round(np.max(gain), 6),
        f"{prefix}_mean.gain": np.round(np.mean(gain), 6),
        f"{prefix}_min.gain": np.round(np.min(gain), 6),
        f"{prefix}_std.gain": np.round(np.std(gain), 6),
    }


def load_hyperparameters(file: str, n_genomes=None):
    with open(file, "r") as f:
        data: dict = json.load(f)

        if n_genomes:
            data.update({"n.weights": n_genomes})

        return data


# Hyper parameters
hyper = load_hyperparameters(file="hyperparameters.json", n_genomes=n_network_weights)

# tuer
tuner = Tuner(hyperparameters=hyper)

# log file
headers = [
    "set of enemies  ",
    "run id",
    "gen",
    "train_max.fitness",
    "train_mean.fitness",
    "train_min.fitness",
    "train_std.fitness",
    "train_max.gain",
    "train_mean.gain",
    "train_min.gain",
    "train_std.gain",
    "test_max.fitness",
    "test_mean.fitness",
    "test_min.fitness",
    "test_std.fitness",
    "test_max.gain",
    "test_mean.gain",
    "test_min.gain",
    "test_std.gain",
]

headers_for_printing = [
    "set of enemies  ",
    "run id",
    "gen",
    "train_max.fitness",
    "train_max.gain",
    "train_mean.fitness",
    "train_mean.gain",
]

logger = Logger(headers=headers, headers_for_printing=headers_for_printing, print=True)

with open("logs.txt", "w") as f:
    f.write(",".join([str(x).strip() for x in logger.headers]) + "\n")

pairs_list = [[6, 8], [1, 8], [4, 7], [1, 2], [3, 5], [2, 3], [3, 4], [7, 8], [1, 4], [2, 7]]
trios_list = [[2, 6, 8], [2, 8, 7], [1, 5, 6], [1, 5, 8], [3, 5, 6], [5, 6, 8], [3, 5, 8], [1, 6, 8], [1, 4, 8], [1, 2, 5]]
groups_of_4_list = [[2, 3, 4, 7], [2, 3, 6, 8], [2, 3, 7, 8], [1, 2, 6, 8], [1, 2, 6, 7], [1, 3, 4, 5], [5, 6, 7, 8], [2, 3, 5, 6], [1, 3, 4, 7], [1, 2, 5, 7]]

#  Training
enemy_set = groups_of_4_list
run_p_enemy = 5

for enemies in enemy_set:

    train_env = Environment(
        experiment_name="./",
        multiplemode="yes",
        enemies=enemies,
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False,
    )

    for run in range(run_p_enemy):

        algo = Ga()

        print(2 * "\n" + 7 * "-" + f" run {run}, for set {enemies} " + 7 * "-", end="\n\n")
        print(2 * "\n" + 7 * "-" + " Start Evolving " + 7 * "-", end="\n\n")

        logger.print_headers()

        # data log
        log = {h: 0 for h in headers}

        # Initialize population
        population_w = algo.initialize_population(population_size=hyper["population.size"], n_genomes=hyper["n.weights"], normal=False)
        population_f, population_g = evaluate(population_w, train_env)
        t_population_f, t_population_g = evaluate(population_w, test_env)

        log.update(
            {
                "set of enemies  ": " ".join(str(e) for e in train_env.enemies),
                "run id": dt.datetime.today().strftime("%M:%S"),
                "gen": 0,
                **calc_statistics(fitness=population_f, gain=population_g, prefix="train"),
                **calc_statistics(fitness=t_population_f, gain=t_population_g, prefix="test"),
            },
        )

        logger.save_log(log)  # print values

        # Start Evolving
        for generation in range(1, hyper["n.generations"] + 1):

            # PARENT SELECTION
            parents_w, parents_f = algo.tournament_selection(population_w, population_f, hyper["tournament.size"])

            # CROSSOVER
            offspring_w = algo.crossover_n_offspring(parents_w, hyper["n.offspring"])

            # MUTATION
            offspring_w = algo.mutate(offspring=offspring_w, p_mutation=hyper["p.mutate.individual"], p_genome=hyper["p.mutate.genome"], sigma_mutation=hyper["sigma.mutate"])

            # RE-EVALUATE
            combined_w = np.vstack((population_w, offspring_w))
            combined_f, combined_g = evaluate(combined_w, train_env)

            # SURVIVAL SELECTION
            population_w, population_f = algo.take_survivors(combined_w, combined_f, size=hyper["population.size"])
            t_population_f, t_population_g = evaluate(population_w, test_env)

            log.update(
                {
                    "gen": generation,
                    **calc_statistics(fitness=population_f, gain=population_g, prefix="train"),
                    **calc_statistics(fitness=t_population_f, gain=t_population_g, prefix="test"),
                }
            )

            logger.save_log(log)

            # some_threshold = 0.01
            # dist = tuner.similairWeights(population_w)
            # close_by = np.sum(dist < some_threshold, axis=1)
            # print(f"\ngeneration: {generation}")
            # print(f"fitness of population: \n {population_f}")
            # print(f"close by neighbour:\n {close_by}")

        print(2 * "\n" + 7 * "-" + " Finished Evolving " + 7 * "-", end="\n\n")

        # write run to file
        with open("logs.txt", "a") as f:
            log_length = max(len(values) for values in logger.logs.values())
            for i in range(log_length):
                f.write(",".join([str(logger.logs[key][i]) for key in logger.logs.keys()]) + "\n")

        logger.clean_logs()
