# Play with winner
import os
import random
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller

# Define the file path
file_path = "best_weights_8_benchmark.txt_test"

# Read solution
solution = []

with open(file_path, "r") as f:
    lines = f.readlines()
    solution.append(lines)
    

# Setup environment
env = Environment(experiment_name="testrun",
                  playermode="ai",
                  player_controller=player_controller(10),
                  speed="normal",
                  enemymode="static",
                  level=2,
                  visuals=True)

def simulation(x, env):
    #f, p, e, t = env.play(pcont=x)

    per_enemy_stats, average_stats = env.play(pcont=x)
    f = average_stats.avg_fitness
    p = average_stats.avg_player_life
    e = average_stats.avg_enemy_life
    t = average_stats.avg_time
    
    return per_enemy_stats, average_stats

def evaluate(x, env):
    evaluation = np.array(list(map(lambda y: simulation(y, env), x)))
    return evaluation[:, 0], evaluation[:, 1]

per_enemy_stats, average_stats = evaluate([solution], env)

print(per_enemy_stats)
print(average_stats)

#env.play(solution)
