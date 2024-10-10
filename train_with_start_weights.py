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

# seed
np.random.seed(420)

# notebook settings
settings = {
    "showTestRun": False,  # Show the training afterwards
    "saveLogs": True,  # Save the logs to a file named logs
    "logfile": "./logs.txt",  # where to save the logs
    "saveWeights": True,  # Save the weights to a file named weights
    "weightsfile": "./weights.txt",  # where to save the weights
    "saveWeightLogs": True, # Save weight changing gain results
    "weightLogfile": "./weightLogs.txt",  # where to save the weightlogs
}

if not settings["showTestRun"]:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

##############################
##### Initialize Environment
##############################


# Environment Settings
n_hidden_neurons = 10
n_network_weights = (20 + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5 # 265

enemies = [1,2,3,4,5,6,7,8]
                                    # Should equal lenght of 'enemies', and sum to 1

########################
## Initialize Weights ##
########################

WEIGHTS = [[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
           [0.500, 0.071428, 0.071428, 0.071428, 0.071428, 0.071428, 0.071428, 0.071428],
           [0.1, 0.1, 0.1, 0.1, 0.1, 0.20, 0.15, 0.15],
           [0.3, 0.1, 0.1, 0.1, 0.05, 0.05, 0.15, 0.15],
           [0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2],
           [0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.25, 0.25],
           [0.15, 0.2, 0.05, 0.1, 0.1, 0.15, 0.15, 0.1],
           [0.05, 0.1, 0.2, 0.25, 0.1, 0.15, 0.1, 0.05],
           [0, 0.05, 0.2, 0.25, 0.25, 0.1, 0.05, 0.1],
           [0.125, 0.175, 0.1, 0.075, 0.2, 0, .2, 0.125]]

def check_weights(enemies, weights):
    for i, weight_list in enumerate(weights):
        # Ensure weights and enemies have the same length
        if len(weight_list) != len(enemies):
            raise ValueError("Length of weights and values must match.")
        # Ensure weights sum up to 1
        if not np.isclose(np.sum(weight_list), 1.0, atol=1e-3):
            raise ValueError(f"Sum of weight values must equal 1, but for list {i} is {np.sum(weight_list)}")

check_weights(enemies, WEIGHTS)

#           [[0.25,0.25,0.25,0.25],
#            [0.35,0.30,0.15,0.20],
#            [0.30,0.15,0.20,0.35],
#            [0.15,0.20,0.35,0.30],
#            [0.20,0.35,0.30,0.15],
#            [0.10,0.10,0.40,0.40],
#            [0.10,0.40,0.40,0.10],
#            [0.40,0.10,0.10,0.40],
#            [0.10,0.40,0.10,0.40],
#            [0.60,0.10,0.10,0.20],
#            [0.10,0.60,0.20,0.10],
#            [0.10,0.20,0.10,0.60]]

# Funtion that updates weights based on enemy gain stats
def update_weights(per_enemy_stats, weights, increase_factor=0.1):
    # Calculate gains (player_life - enemy_life) for each enemy
    gains = np.array([stats.player_life - stats.enemy_life for stats in per_enemy_stats])

    # Find index of the enemy with the lowest gain
    lowest_gain_idx = np.argmin(gains)

    # Update the weight of the lowest gain enemy
    new_weights = np.array(weights)
    new_weights[lowest_gain_idx] += increase_factor

    # Adjust the remaining weights so that the total sum is 1
    excess = new_weights[lowest_gain_idx] - weights[lowest_gain_idx]
    remaining_sum = 1 - new_weights[lowest_gain_idx]

    # Distribute the remaining weights proportionally among the other enemies
    for i in range(len(weights)):
        if i != lowest_gain_idx:
            new_weights[i] *= remaining_sum / (1 - weights[lowest_gain_idx])

    # Normalize weights to ensure the sum is exactly 1 (in case of floating-point precision issues)
    new_weights /= new_weights.sum()

    return new_weights


##############################
##### Initialize Data Tracking
##############################


def calc_statistics(fitness: np.array):
    return {
        "max.fitness": np.round(np.max(fitness), 4),
        "mean.fitness": np.round(np.mean(fitness), 4),
        "min.fitness": np.round(np.min(fitness), 4),
        "std.fitness": np.round(np.std(fitness), 4),
    }


headers = ["run id", "generation", "max.fitness", "mean.fitness", "min.fitness", "std.fitness", "set of enemies  "]
logger = Logger(headers=headers)

weights_headers = ["run_id", "enemy_weights", "enemy_gains", "avg_gain"]
weights_logger = Logger(headers=weights_headers)

##############################
##### Initialize Hyper parameters
##############################


def load_hyperparameters(file: str, n_genomes=None):
    with open(file, "r") as f:
        data = json.load(f)

        if n_genomes:
            data.update({"n_genomes": n_genomes})

        return data


hyper = load_hyperparameters(file="hyperparameters.json", n_genomes=n_network_weights)



##############################
##### Tuner
##############################

tuner = Tuner(hyperparameters=hyper)

##############################
##### Start Simulation
##############################

algo = Ga()

nr_weight_tests = 0
MAX_NR_WEIGHT_TESTS = 4

STARTING_WEIGHTS = [[0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.25, 0.25]] # More weight on 7,8
check_weights(enemies, STARTING_WEIGHTS)
enemy_weights_results = []

# Start loop for finding good enemy_weights
enemy_weights = STARTING_WEIGHTS[0]

while (nr_weight_tests <= MAX_NR_WEIGHT_TESTS): # Stop after search limit reached OR ...            #for i, weight_list in enumerate(WEIGHTS):
    env = Environment(
        experiment_name="./",
        multiplemode="yes",
        enemies=enemies,
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False,
        weights=enemy_weights,
        use_weights=True,
    )
    
    def simulation(x):
        per_enemy_stats, average_stats = env.play(pcont=x)
        #f, p, e, t = env.play(pcont=x)
        return average_stats.avg_fitness


    def evaluate(x):
        return np.array(list(map(lambda y: simulation(y), x)))

    print(2 * "\n" + 7 * "-" + f" run {nr_weight_tests} " + 7 * "-", end="\n\n")
    print(2 * "\n" + 7 * "-" + " Start Evolving " + 7 * "-", end="\n\n")
    # logger.print_headers()

    run_best_w, run_best_f = [], -100

    # data log
    log = {h: 0 for h in headers}

    # Initialize population
    population_w = algo.initialize_population(population_size=hyper["population.size"], n_genomes=hyper["n.weights"])
    population_f = evaluate(population_w)

    log.update(
        {
            "run id": str(f"{nr_weight_tests}_" + dt.datetime.today().strftime("%H:%M")),
            "generation": 0,
            "set of enemies  ": " ".join(str(e) for e in env.enemies),
            **calc_statistics(population_f),
        },
    )

    logger.save_log(log)  # print values

    # Start Evolving
    for generation in range(1, hyper["n.generations"] + 1):

        # PARENT SELECTION
        parents_w, parents_f = algo.tournament_selection(population_w, population_f, hyper["tournament.size"])

        # CROSSOVER
        offspring_w = algo.crossover_2_offspring(parents_w, p=hyper["p.reproduce"])

        # MUTATION
        offspring_w = algo.mutate(offspring=offspring_w, p_mutation=hyper["p.mutate.individual"], p_genome=hyper["p.mutate.genome"], sigma_mutation=hyper["sigma.mutate"])

        # EVALUATE
        offspring_f = evaluate(offspring_w)

        n_best_w, n_best_f = np.empty([hyper["n.best"], hyper["n.weights"]]), np.empty([hyper["n.best"], 1])

        if hyper["n.best"] > 0:
            n_best_w, n_best_f, population_w, population_f = algo.eletist_selection(population_w, population_f, hyper["n.best"])

        # COMBINE
        combined_w = np.vstack((population_w, offspring_w))
        combined_f = np.append(population_f, offspring_f)

        selected_w, selected_f = algo.survival_selection(combined_w, combined_f, hyper["population.size"] - hyper["n.best"])

        population_w = np.vstack((selected_w, n_best_w))
        population_f = np.append(selected_f, n_best_f)

        best_idx = np.argmax(population_f)
        run_best_w = population_w[best_idx]
        run_best_f = population_f[best_idx]

        log.update({"generation": generation, **calc_statistics(population_f)})

        logger.save_log(log)

        # Apply tuning Logic

        # diversity = 0
        # for i in range(population_w.shape[1]):
        #     diversity += np.std(population_w[:, i])
        # print("DIVERSITY:", diversity)

        # lookback = 3

        # if len(logger.logs["max.fitness"]) >= lookback and tuner.readyforupdate(generation, lookback):

        #     if tuner.hasProgressed(name="max.fitness", metrics=logger.logs["max.fitness"], lookback=lookback, threshold=10):
        #         new_val = np.max([hyper["sigma.mutate"] - 0.10, 0.1])
        #         hyper = tuner.updateHyperparameter(key="sigma.mutate", value=new_val, generation=generation, lookback=lookback)

        #     else:
        #         new_val = np.min([hyper["sigma.mutate"] + 0.10, 1.5])
        #         hyper = tuner.updateHyperparameter(key="sigma.mutate", value=new_val, generation=generation, lookback=lookback)

        #     if tuner.diversity_low(population_w, 160):
        #         new_val = hyper["p.reproduce"] + 0.1
        #         hyper = tuner.updateHyperparameter(key="p.reproduce", value=new_val, generation=generation, lookback=lookback)

        #     else:
        #         new_val = hyper["p.reproduce"] - 0.1
        #         hyper = tuner.updateHyperparameter(key="p.reproduce", value=new_val, generation=generation, lookback=lookback)

    print(2 * "\n" + 7 * "-" + " Finished Evolving " + 7 * "-", end="\n\n")

    
    # Show Test Result
    env.update_parameter("enemies", [1, 2, 3, 4, 5, 6, 7, 8])
    env.update_parameter("use_weights", False,)
    
    # Get gain results per enemy and overall
    per_enemy_stats, average_stats = env.play(run_best_w)

    f = average_stats.avg_fitness
    p = average_stats.avg_player_life
    e = average_stats.avg_enemy_life
    t = average_stats.avg_time

    # Store weight results
    enemy_gains = list([stats.player_life - stats.enemy_life for stats in per_enemy_stats])
    avg_gain = p-e
    enemy_weights_results.append((enemy_weights, enemy_gains, avg_gain))

    weights_log = {h: 0 for h in weights_headers}
    weights_log.update(
        {"run id": str(f"{nr_weight_tests-1}_" + dt.datetime.today().strftime("%H:%M")),
        "enemy_weights": enemy_weights,
        "enemy_gains": enemy_gains,
        "avg_gain" : avg_gain},
    )

    weights_logger.save_log(weights_log)

    # Update weights based on enemy effectiveness
    new_weights = update_weights(per_enemy_stats, enemy_weights, increase_factor=0.05)
    enemy_weights = new_weights
    nr_weight_tests += 1

    # print outcome of training
    print(f"\nAFTER TESTING RUN ON {env.enemies}\n")
    outcome = Logger(["avg.fitness", "avg.playerlife", "avg.enemylife", "avg.time", "avg.gain"])
    outcome.print_headers()
    outcome.print_log([np.round(x, 2) for x in [f, p, e, t, p - e]])

    logger.save_log(log) 


print(enemy_weights_results)

##############################
##### Post Simulation
##############################


if settings["saveLogs"]:
    printHeaders = True if not os.path.exists(settings["logfile"]) else False

    # Write to file
    with open(settings["logfile"], "a+") as f:

        log_length = max(len(values) for values in logger.logs.values())

        if printHeaders:
            f.write(",".join([str(x) for x in logger.headers]) + "\n")

        for i in range(log_length):
            line = [str(logger.logs[key][i]) for key in logger.logs.keys()]
            f.write(",".join(line) + "\n")

if settings["saveWeightLogs"]:
    printHeaders = True if not os.path.exists(settings["weightLogfile"]) else False

    # Write to file
    with open(settings["weightLogfile"], "a+") as f:

        log_length = max(len(values) for values in weights_logger.logs.values())

        if printHeaders:
            f.write(",".join([str(x) for x in weights_logger.headers]) + "\n")

        for i in range(log_length):
            line = [str(weights_logger.logs[key][i]) for key in weights_logger.logs.keys()]
            f.write(",".join(line) + "\n")


# if settings["saveWeights"]:
#     with open(settings["weightsfile"], "w") as f:
#         f.write("\n".join([str(x) for x in run_best_w]))

# # Show Test Run
# if settings["showTestRun"]:
#     env.update_parameter("speed", "normal")
#     env.update_parameter("visuals", "True")
