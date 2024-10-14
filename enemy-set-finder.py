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
    #f, p, e, t = env.play(pcont=x)

    per_enemy_stats, average_stats = env.play(pcont=x)
    f = average_stats.avg_fitness
    p = average_stats.avg_player_life
    e = average_stats.avg_enemy_life
    t = average_stats.avg_time
    
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
    "set of enemies",
    "enemy_weights",
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
    "set of enemies",
    "enemy_weights",
    "run id",
    "gen",
    "train_max.fitness",
    "train_max.gain",
    "test_max.fitness",
    "test_max.gain",
]

logger = Logger(headers=headers, headers_for_printing=headers_for_printing, print=False)

with open("logs.txt", "w") as f:
    f.write(",".join([str(x).strip() for x in logger.headers]) + "\n")


#### INITIALIZE TRAINING RUN SETTINGS #######
groups_of_4_list = [[2, 3, 6, 8]]
groups_of_8_list = [[1,2,3,4,5,6,7,8]]


# Should equal lenght of 'enemies', and sum to 1
WEIGHTS_4 = [[0.25,0.25,0.25,0.25],      # Equal weighting for benchmark
           [0.40,0.40,0.10,0.10],        # More weight on : 2,3  , Less weight on : 6,8
           [0.10,0.10,0.40,0.40],]       # More weight on : 6,8  , Less weight on : 2,3

WEIGHTS_8 = [[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],     # Equal weighting for benchmark
             [0.05, 0.20, 0.20, 0.05, 0.05, 0.20, 0.05, 0.20],
             [0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.0, 0.25], 
        #    [0.10, 0.10, 0.10, 0.10, 0.05, 0.25, 0.05, 0.25],                # More weight on : 6 8,    Less weight on : 5 7
        #    [0.10, 0.05, 0.05, 0.10, 0.10, 0.25, 0.10, 0.25],                # More weight on : 6 8,    Less weight on : 4 3
        #    [0.05, 0.10, 0.05, 0.10, 0.10, 0.20, 0.20, 0.20],                # More weight on : 6 7 8, Less weight on : 1 3
        #    [0.05, 0.15, 0.15, 0.05, 0.15, 0.15, 0.15, 0.15],                # More weight on : -    , Less weight on : 1 4
        #    [0.02, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],                # More weight on : -   , Less weight on : 1
        #    [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.02, 0.14]                # More weight on : -  , Less weight on : 7
]
def check_weights(enemies, weights):
    for i, weight_list in enumerate(weights):
        # Ensure weights and enemies have the same length
        if len(weight_list) != len(enemies):
            raise ValueError("Length of weights and values must match.")
        # Ensure weights sum up to 1
        if not np.isclose(np.sum(weight_list), 1.0, atol=1e-3):
            raise ValueError(f"Sum of weight values must equal 1, but for list {i} is {np.sum(weight_list)}")

check_weights(groups_of_4_list[0], WEIGHTS_4)
check_weights(groups_of_8_list[0], WEIGHTS_8)

# Define best_weights file
best_weights_file = "best_weights_8_benchmark.txt"

# Check if the file exists
if os.path.exists(best_weights_file):
    # If the file exists, read the first line (which should contain the stored max_gain)
    with open(best_weights_file, "r") as f:
        stored_max_gain = float(f.readline().strip())
else:
    with open(best_weights_file, "r") as f: # Set to a very low number if no file exists
        stored_max_gain = float(f.readline().strip())  
        f.write(str(stored_max_gain) + "\n")


#### START TRAINING #######
enemy_set = groups_of_8_list
enemy_weights_set = WEIGHTS_8
run_p_enemy = 3

np.random.seed(420)

for enemies in enemy_set:

    for enemy_weights in enemy_weights_set:

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
            enemy_weights=enemy_weights,
            use_enemy_weights=True,
        )

        for run in range(run_p_enemy):

            algo = Ga()

            print(2 * "\n" + 7 * "-" + f" run {run}, for set {enemies}, for weights {enemy_weights}" + 7 * "-", end="\n\n")
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
                    "set of enemies": " ".join(str(e) for e in enemies),
                    "enemy_weights": " ".join(str(e) for e in enemy_weights),
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
                # parents_w, parents_f = algo.tournament_selection(population_w, population_g, hyper["tournament.size"])

                # CROSSOVER
                offspring_w = algo.crossover_n_offspring(parents_w, hyper["n.offspring"])

                # MUTATION
                offspring_w = algo.mutate(offspring=offspring_w, p_mutation=hyper["p.mutate.individual"], p_genome=hyper["p.mutate.genome"], sigma_mutation=hyper["sigma.mutate"])

                # RE-EVALUATE
                combined_w = np.vstack((population_w, offspring_w))
                combined_f, combined_g = evaluate(combined_w, train_env)

                # SURVIVAL SELECTION
                population_w, population_f = algo.take_survivors(combined_w, combined_f, size=hyper["population.size"])
                # population_w, population_f = algo.take_survivors(combined_w, combined_g, size=hyper["population.size"])
                
                log.update(
                    {
                        "gen": generation,
                        **calc_statistics(fitness=population_f, gain=population_g, prefix="train"),
                        **calc_statistics(fitness=np.array([0]), gain=np.array([0]), prefix="test")
                    }
                )

                logger.save_log(log)

                # some_threshold = 0.01
                # dist = tuner.similairWeights(population_w)
                # close_by = np.sum(dist < some_threshold, axis=1)
                # print(f"\ngeneration: {generation}")
                # print(f"fitness of population: \n {population_f}")
                # print(f"close by neighbour:\n {close_by}")

            
            # Validate on test set            
            t_population_f, t_population_g = evaluate(population_w, test_env)
            
            # Store weights of the individual with the max gain after the last generation
            max_gain = np.max(t_population_g)
            max_gain_index = np.argmax(t_population_g)  # Index of the individual with max gain
            best_individual_weights = population_w[max_gain_index]
            
            # If the current max_gain_all_gens is greater than the stored one, overwrite the file
            if max_gain > stored_max_gain:
                stored_max_gain = max_gain

                with open(best_weights_file, "w") as f: # Opens file and clears content immediately
                    # Write the max_gain as the first line
                    f.write(f"{max_gain}\n")

                    # Write each weight on a new line
                    for weight in best_individual_weights:  
                        f.write(f"{weight}\n")

            
            log.update(
                    {
                        "gen": "validation",
                        **calc_statistics(fitness=np.array([0]), gain=np.array([0]), prefix="train"),
                        **calc_statistics(fitness=t_population_f, gain=t_population_g, prefix="test"),
                    }
            )
            logger.save_log(log)


            print(2 * "\n" + 7 * "-" + " Finished Evolving " + 7 * "-", end="\n\n")

            # write run to file
            with open("logs.txt", "a") as f:
                log_length = max(len(values) for values in logger.logs.values())
                for i in range(log_length):
                    f.write(",".join([str(logger.logs[key][i]) for key in logger.logs.keys()]) + "\n")

            # clean logs, set only header values
            logger.clean_logs()