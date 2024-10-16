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

# notebook settings
settings = {
    "showTestRun": False,  # Show the training afterwards
}


##############################
##### Initialize Environment
##############################

# Environment Settings
enemies = [2, 5, 6, 8]
# enemies = [1, 2, 3, 4, 5, 6, 7, 8]

n_hidden_neurons = 10
n_network_weights = (20 + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5


if not settings["showTestRun"]:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def simulation(x, env):
    # f = fitness, p = playerlife, e = enemylife, t = time, w = wins, ng = sum(gained), g = avg(gain) - std(gain)
    f, p, e, t, w, ng, g = env.play(pcont=x)
    return f, ng, w


def evaluate(x, env):
    evaluation = np.array(list(map(lambda y: simulation(y, env), x)))
    return evaluation[:, 0], evaluation[:, 1], evaluation[:, 2]


def load_hyperparameters(file: str, n_genomes=None):
    with open(file, "r") as f:
        data: dict = json.load(f)

        if n_genomes:
            data.update({"n.weights": n_genomes})

        return data


# Hyper parameters
hyper = load_hyperparameters(file="hyperparameters.json", n_genomes=n_network_weights)
hyper_defaults = hyper.copy()

# tuner
tuner = Tuner(hyperparameters=hyper)

# log file
headers = [
    "set of enemies  ",
    "run id",
    "gen",
    #
    "champ.fitness",
    "champ.gain",
    "champ.wins",
    #
    "train_mean.wins",
    "train_mean.fitness",
    "train_std.fitness",
    "train_mean.gain",
    "train_std.gain",
    #
    "genotype.dist",
    "p.mutate",
    "sigma",
    "offspring",
    "phase",
]

# print outcome of trainign
outcome_headers = ["wins", "sum(gain)", "avg(gain)", "gain", "fitness", "Playerlife", "Enemylife"]

outcome = Logger(headers=outcome_headers)
logger = Logger(headers=headers)


algo = Ga()

for run in range(5):

    hyper = hyper_defaults.copy()

    phase = "fixed parameters"
    free_period = 15  # period before tuning starts
    explore_time = 10  # how many evolutions does the algo get to test new parameters settings

    # updates
    update_ts = 0
    update_step = 0
    updates = [(0.35, 0.35, 4), (0.70, 0.70, 6), (0.7, 1.4, 8)]

    with open(f"logfile_{run}.txt", "w") as f:
        f.write(",".join([str(x) for x in logger.headers]) + "\n")

    with open(f"outcomesfile_{run}.txt", "w") as f:
        f.write(",".join([str(x) for x in outcome.headers]) + "\n")

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

    print(2 * "\n" + 7 * "-" + f" Start Evolving / run {run} " + 7 * "-", end="\n\n")
    logger.print_headers()

    run_best_w, run_best_f = [], float("-inf")

    # data log
    log = {h: 0 for h in headers}

    # Initialize population
    population_w = algo.initialize_population(population_size=hyper["population.size"], n_genomes=hyper["n.weights"], normal=True)
    population_f, population_g, population_v = evaluate(population_w, env)

    log.update(
        {
            "set of enemies  ": " ".join(str(e) for e in env.enemies),
            "run id": dt.datetime.today().strftime("%M:%S"),
            "gen": 0,
            "champ.fitness": 0,
            "champ.gain": 0,
            "champ.wins": 0,
            "train_mean.wins": np.round(np.mean(population_v), 6),
            "train_mean.fitness": np.round(np.mean(population_f), 6),
            "train_std.fitness": np.round(np.std(population_f), 6),
            "train_mean.gain": np.round(np.mean(population_g), 6),
            "train_std.gain": np.round(np.std(population_g), 6),
            "genotype.dist": np.round(np.mean(tuner.similairWeights(population_w)), 6),
            "p.mutate": hyper["p.mutate.genome"],
            "sigma": hyper["sigma.mutate"],
            "offspring": hyper["n.offspring"],
            "phase": phase,
        },
    )

    logger.save_log(log)

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
        combined_f, combined_g, combined_v = evaluate(combined_w, env)

        # SURVIVAL SELECTION
        population_w, population_f, population_g, population_v = algo.select_survivors_multi_object(combined_w, combined_f, combined_g, combined_v, size=hyper["population.size"])

        # SAVE BEST
        run_best_w = population_w[-1]
        run_best_f = population_f[-1]
        run_best_g = population_g[-1]
        run_best_v = population_v[-1]

        log.update(
            {
                "gen": generation,
                "champ.fitness": np.round(run_best_f, 6),
                "champ.gain": np.round(run_best_g, 6),
                "champ.wins": np.round(run_best_v, 6),
                "train_mean.wins": np.round(np.mean(population_v), 6),
                "train_mean.fitness": np.round(np.mean(population_f), 6),
                "train_std.fitness": np.round(np.std(population_f), 6),
                "train_mean.gain": np.round(np.mean(population_g), 6),
                "train_std.gain": np.round(np.std(population_g), 6),
                "genotype.dist": np.round(np.mean(tuner.similairWeights(population_w)), 6),
                "p.mutate": hyper["p.mutate.genome"],
                "sigma": hyper["sigma.mutate"],
                "offspring": hyper["n.offspring"],
                "phase": phase,
            }
        )

        logger.save_log(log)

        # # TUNER
        # if generation >= free_period:
        #     if generation - update_ts >= explore_time:

        #         if tuner.noProgress(logger.get("champ.gain"), threshold=0, lookback=13):
        #             phase = "Exploring"
        #             update_ts = generation
        #             update_step += 1

        #             if update_step >= 3:
        #                 update_step = 0
        #                 population_w = algo.repopulate(population_w, frac=0.8)
        #                 population_f, population_g, population_v = evaluate(population_w, env)
        #                 hyper.update({"p.mutate.genome": updates[update_step][0]})
        #                 hyper.update({"sigma.mutate": updates[update_step][1]})
        #                 hyper.update({"n.offspring": updates[update_step][2]})
        #             else:
        #                 hyper.update({"p.mutate.genome": updates[update_step][0]})
        #                 hyper.update({"sigma.mutate": updates[update_step][1]})
        #                 hyper.update({"n.offspring": updates[update_step][2]})

        #         else:
        #             phase = "Exploiting"
        #             update_step = 0
        #             hyper.update({"p.mutate.genome": updates[update_step][0]})
        #             hyper.update({"sigma.mutate": updates[update_step][1]})
        #             hyper.update({"n.offspring": updates[update_step][2]})

    print(2 * "\n" + 7 * "-" + " Finished Evolving " + 7 * "-", end="\n\n")

    # Show Test Result
    env.update_parameter("enemies", [1, 2, 3, 4, 5, 6, 7, 8])
    f, p, e, t, w, ng, g = env.play(run_best_w)

    with open(f"outcomesfile_{run}.txt", "a") as file:
        file.write(",".join([str(np.round(x, 2)) for x in [w, ng, g, p - e, f, p, e]]))

    with open(f"weightsfile_{run}.txt", "w") as f:
        f.write("\n".join([str(x) for x in run_best_w]))

    with open(f"logfile_{run}.txt", "a") as f:
        log_length = max(len(values) for values in logger.logs.values())

        for i in range(log_length):
            line = [str(logger.logs[key][i]) for key in logger.logs.keys()]
            f.write(",".join(line) + "\n")
