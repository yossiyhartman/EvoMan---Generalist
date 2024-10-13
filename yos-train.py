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
    "showTestRun": True,  # Show the training afterwards
    "saveLogs": False,  # Save the logs to a file named logs
    "printLogs": True,
    "logfile": "./logs.txt",  # where to save the logs
    "saveWeights": False,  # Save the weights to a file named weights
    "weightsfile": "./weights.txt",  # where to save the weights
}


##############################
##### Initialize Environment
##############################

# Environment Settings
n_hidden_neurons = 10
n_network_weights = (20 + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

enemies = [6, 8]


if not settings["showTestRun"]:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


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
)


def simulation(x, gain=False):
    f, p, e, t = env.play(pcont=x)
    return p - e


def evaluate(x):
    return np.array(list(map(lambda y: simulation(y), x)))


##############################
##### Initialize Data Tracking
##############################


def calc_statistics(fitness: np.array):
    return {
        "max.gain": np.round(np.max(fitness), 3),
        "mean.gain": np.round(np.mean(fitness), 3),
        "min.gain": np.round(np.min(fitness), 3),
        "std.gain": np.round(np.std(fitness), 3),
    }


headers = ["set of enemies  ", "run id", "gen", "max.gain", "mean.gain", "min.gain", "std.gain", "genotype.dist", "p.indivi", "p.genome", "simga", "tour.size"]
logger = Logger(headers=headers, print=settings["printLogs"])

##############################
##### Initialize Hyper parameters
##############################


def load_hyperparameters(file: str, n_genomes=None):
    with open(file, "r") as f:
        data: dict = json.load(f)

        if n_genomes:
            data.update({"n_genomes": n_genomes})

        return data


hyper_defaults = load_hyperparameters(file="hyperparameters.json", n_genomes=n_network_weights)
hyper = hyper_defaults.copy()

h = Logger(headers=hyper.keys(), print=settings["printLogs"])
h.print_headers()
h.print_log(hyper.values())

##############################
##### Tuner
##############################

tuner = Tuner(hyperparameters=hyper)

free_period = 10  # period before tuning starts
explore_time = 7  # how many evolutions does the algo get to test new parameters settings
lookback = 5  # comparison window

update_timestamp = {k: 0 for k in hyper.keys()}
pivot_to_exploration_ts = 0

##############################
##### Start Simulation
##############################

algo = Ga()

for _ in range(1):

    print(2 * "\n" + 7 * "-" + f" run {_} " + 7 * "-", end="\n\n")
    print(2 * "\n" + 7 * "-" + " Start Evolving " + 7 * "-", end="\n\n")
    logger.print_headers()

    run_best_w, run_best_f = [], -100

    # data log
    log = {h: 0 for h in headers}

    # Initialize population
    population_w = algo.initialize_population(population_size=hyper["population.size"], n_genomes=hyper["n.weights"])
    population_f = evaluate(population_w)

    log.update(
        {
            "run id": dt.datetime.today().strftime("%M:%S"),
            "gen": 0,
            "set of enemies  ": " ".join(str(e) for e in env.enemies),
            **calc_statistics(population_f),
            "genotype.dist": np.round(tuner.similairWeights(population_w), 3),
            "p.indivi": hyper["p.mutate.individual"],
            "p.genome": hyper["p.mutate.genome"],
            "simga": hyper["sigma.mutate"],
            "tour.size": hyper["tournament.size"],
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

        log.update(
            {
                "gen": generation,
                **calc_statistics(population_f),
                "genotype.dist": np.round(tuner.similairWeights(population_w), 3),
                "p.indivi": hyper["p.mutate.individual"],
                "p.genome": hyper["p.mutate.genome"],
                "simga": hyper["sigma.mutate"],
                "tour.size": hyper["tournament.size"],
            }
        )

        logger.save_log(log)

        # Tuning

        if generation >= free_period:

            ### per metric

            # per metric
            # can_update = {k: (generation - update_timestamp[k] >= explore_time) for k in update_timestamp.keys()}

            # # check when the latest update is done
            # if can_update["sigma.mutate"]:

            #     if tuner.noMaxIncrease(logger.logs["max.gain"], 0, lookback):
            #         new_sigma = np.min([hyper["sigma.mutate"] + 0.10, 0.6])
            #         hyper.update({"sigma.mutate": new_sigma})
            #         update_timestamp.update({"sigma.mutate": generation})
            #     else:
            #         hyper = hyper_defaults

            # combined update function

            ### global

            # check when the latest update is done
            if generation - pivot_to_exploration_ts >= explore_time:

                if tuner.noMaxIncrease(logger.logs["max.gain"], 0, lookback):
                    update = {
                        "p.mutate.individual": np.round(np.min([hyper["p.mutate.individual"] + 0.20, 0.5]), 3),
                        "p.mutate.genome": np.round(np.min([hyper["p.mutate.genome"] + 0.20, 0.5]), 3),
                        "sigma.mutate": np.round(np.min([hyper["sigma.mutate"] + 0.20, 0.6]), 3),
                        "tournament.size": np.round(np.min([hyper["tournament.size"] + 4, 10]), 3),
                    }
                    hyper.update(**update)
                    pivot_to_exploration_ts = generation
                else:
                    hyper = hyper_defaults

                if tuner.noMeanMaxDifference(max_fitness=logger.logs["max.gain"], mean_fitness=logger.logs["mean.gain"], threshold=5, lookback=lookback):
                    population_w = algo.repopulate(population_w, population_f, frac=0.9)
                    population_f = evaluate(population_w)

    print(2 * "\n" + 7 * "-" + " Finished Evolving " + 7 * "-", end="\n\n")


##############################
##### Post Simulation
##############################


if settings["saveLogs"]:
    with open(settings["logfile"], "w") as f:
        f.write(",".join([str(x) for x in logger.headers]) + "\n")

    # Write to file
    with open(settings["logfile"], "a") as f:
        log_length = max(len(values) for values in logger.logs.values())

        for i in range(log_length):
            line = [str(logger.logs[key][i]) for key in logger.logs.keys()]
            f.write(",".join(line) + "\n")


if settings["saveWeights"]:
    with open(settings["weightsfile"], "w") as f:
        f.write("\n".join([str(x) for x in run_best_w]))

# Show Test Run
if settings["showTestRun"]:
    env.update_parameter("speed", "normal")
    env.update_parameter("visuals", "True")

# Show Test Result
env.update_parameter("enemies", [1, 2, 3, 4, 5, 6, 7, 8])
f, p, e, t = env.play(run_best_w)

# print outcome of trainign
outcome = Logger(["avg.gain", "avg.playerlife", "avg.enemylife", "avg.time", "avg.gain"])
outcome.print_headers()
outcome.print_log([np.round(x, 2) for x in [f, p, e, t, p - e]])
