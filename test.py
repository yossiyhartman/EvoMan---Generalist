import json
import os
import datetime as dt
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller

# Our classes
from classes.Logger import Logger

env = Environment(
    experiment_name="./",
    multiplemode="yes",
    enemies=[1, 2, 3, 4, 5, 6, 7, 8],
    playermode="ai",
    player_controller=player_controller(10),
    enemymode="static",
    level=2,
    speed="normal",
    visuals=True,
)


def simulation(x, env):
    # f = fitness, p = playerlife, e = enemylife, t = time, w = wins, ng = sum(gained), g = avg(gain) - std(gain)
    f, p, e, t, w, ng, g = env.play(pcont=x)
    return f, ng, w


def evaluate(x, env):
    evaluation = np.array(list(map(lambda y: simulation(y, env), x)))
    return evaluation[:, 0], evaluation[:, 1], evaluation[:, 2]


run_best_w = []

with open("EA - CHT - 12345678/weightsfile_3.txt", "r", encoding="UTF-8") as f:
    for line in f:
        run_best_w.append(float(line))

run_best_w = np.asarray(run_best_w)


# run_best_w
f, p, e, t, w, ng, g = env.play(run_best_w)

# print outcome of trainign
outcome_headers = ["wins", "sum(gain)", "avg(gain)", "gain", "fitness", "Playerlife", "Enemylife"]
outcome = Logger(headers=outcome_headers)
outcome.print_headers()
outcome.print_log([np.round(x, 2) for x in [w, ng, g, p - e, f, p, e]])
