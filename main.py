"""
Setup a CSTR environment with a setpoint change (see https://maximilianb2.github.io/pc-gym/#welcome)
"""

import time
import numpy as np
import pandas as pd
import pcgym
import matplotlib.pyplot as plt

from models import *
from environment import EnvironmentWrapper
from cnmcts import cnmcts
from cnrpa import run_cnrpa
from random_walk import random_walk
from baselines import get_and_train_ppo

def run_comparison(environment: EnvironmentWrapper, algorithms: dict):
    pass
    scores, execution_times, sequences = dict(), dict(), dict()
    return scores, execution_times, sequences

if __name__ == "__main__":

    # Running algorithms on the CSTR problem
    # Global inputs
    T = 25
    n_steps = 50

    problems = {
        # CSTR problem
        "cstr": {
            "inputs": {
                "problem": "cstr",
                "initial_state": np.array([0.8, 330, 0.8]),
                "set_points": {
                    "Ca": [0.85 for i in range(int(n_steps / 2))]
                    + [0.9 for i in range(int(n_steps / 2))]
            },
            "penalty_factor": np.array([0]),
            "n_steps": n_steps,
            }
        },
        # Crystallization
        "crystallization": {
            "inputs": {
                "problem": "crystallization",
                "initial_state": np.array([1478.01, 22995.82, 1800863.24, 248516167.94, 0.1586, 0.5, 15, 1, 15]),
                "set_points": {
                    "CV": [1 for i in range(int(n_steps / 2))]
                    + [1.1 for i in range(int(n_steps / 2))],
                    "Ln": [15 for i in range(int(n_steps / 2))]
                    + [14 for i in range(int(n_steps / 2))],
                },
            },
            "penalty_factor": np.array([0]),
            "n_steps": n_steps,
        },
        # Four Tank
        "four_tank": {
            "inputs" : {
                "problem": "four_tank",
                "initial_state": np.array([0.141, 0.112, 0.072, 0.42, 0.5, 0.2]),
                "set_points": {
                    "h3": [0.3 for i in range(int(n_steps / 3))]
                    + [0.1 for i in range(int(n_steps / 3))]
                    + [0 for i in range(int(n_steps / 3))],
                    "h4": [0.1 for i in range(int(n_steps / 2))]
                    + [0 for i in range(int(n_steps / 2))],
                },
            },
            "penalty_factor": np.zeros((1, 1)),
            "n_steps": n_steps,
        },
        # Biofilm Reactor
        "biofilm_reactor": {
            "inputs" : {
                "problem": "biofilm_reactor",
                "initial_state": np.array([2,0.1,10,0.1,2,0.1,10,0.1,2,0.1,10,0.1,2,0.1,10,0.1,1]),
                "set_points": {
                    "S1_A": [1 for i in range(int(n_steps / 2))]
                    + [1.1 for i in range(int(n_steps / 2))],
                },
            },
            "penalty_factor": np.zeros((5, 5)),
            "n_steps": n_steps,
        },
        # Example for Nonsmooth control
        "nonsmooth_control": {
            "inputs" : {
                "problem": "nonsmooth_control",
                "initial_state": np.array([0, 0, 0.2]),
                "set_points": {
                    "X1": [0.3 for i in range(int(n_steps / 2))]
                    + [0.4 for i in range(int(n_steps / 2))],
                },
            },
            "penalty_factor": np.zeros((1, 1)),
            "n_steps": n_steps,
        },
        # Example for Multistage Extraction Column
        "multistage_extraction": {
            "inputs": {
                "problem": "multistage_extraction",
                "initial_state": np.array([0.55, 0.3, 0.45, 0.25, 0.4, 0.20, 0.35, 0.15, 0.25, 0.1, 0.3]),
                "set_points": {
                    "X1": [0.2 for i in range(int(n_steps / 2))]
                    + [0.4 for i in range(int(n_steps / 2))],
                    "Y1": [0.2 for i in range(int(n_steps / 2))]
                    + [0.1 for i in range(int(n_steps / 2))],
                },
            },
            "penalty_factor": np.zeros((2, 2)),
            "n_steps": n_steps,
        }
    }
    """

    # Iterating through different parameters
    nesting_levels = {
        1: {"bandwidth": 25, "n_policies": 500, "half_life_divider": 2},
        2: {"bandwidth": 5, "n_policies": 20, "half_life_divider": 2},
    }

    # Enumerating algorithms
    algorithms = dict()
    for level in nesting_levels.keys():
        algorithms["cNMCTS level " + str(level)] = {
            "function": cnmcts,
            "parameters": {
                "level": level,
                "bandwidth": nesting_levels[level]["bandwidth"],
            },
            "plots": [list(), list()],
        }
        algorithms["Gaussian cNRPA level " + str(level)] = {
            "function": run_cnrpa,
            "parameters": {
                "level": level,
                "n_policies": nesting_levels[level]["n_policies"],
                "policy_type": "gaussian",
                "half_life_divider": nesting_levels[level]["half_life_divider"],
            },
            "plots": [list(), list()],
        }
        algorithms["By Region cNRPA level " + str(level)] = {
            "function": run_cnrpa,
            "parameters": {
                "level": level,
                "n_policies": nesting_levels[level]["n_policies"],
                "policy_type": "by_region",
                "half_life_divider": nesting_levels[level]["half_life_divider"],
            },
            "plots": [list(), list()],
        }

    # Number of iterations per algorithm
    n_iterations = 10

    # Storing values
    best_scores = np.zeros((len(penalty_factor_list), len(algorithms) + 1))
    mean_scores = np.zeros((len(penalty_factor_list), len(algorithms) + 1))
    scores_std = np.zeros((len(penalty_factor_list), len(algorithms) + 1))
    average_execution_times = np.zeros((len(penalty_factor_list), len(algorithms) + 1))

    # Inputs
    inputs = {
        "T": T,
        "n_steps": n_steps,
        "initial_state": initial_state,
        "problem": problem,
        "set_points": set_points,
    }
    # Iterating over penalty factors
    for penalty_id, penalty_factor in enumerate(penalty_factor_list):
        print("\n\nPenalty factor:", penalty_factor)
        # Environment
        environment, observation_space = get_environment(**inputs)
        environment = EnvironmentWrapper(environment, n_steps, penalty_factor)

        # Iterating over algorithms
        for algorithm_id, algorithm in enumerate(algorithms.keys()):
            print("\nRunning ", algorithm)
            # Local storage
            scores_list = list()
            times_list = list()

            # Running n_iterations of the same algorithm
            for iteration in range(n_iterations):
                environment.reset()
                start_time = time.time()
                sequence, actions, score = algorithms[algorithm]["function"](
                    environment, **algorithms[algorithm]["parameters"]
                )
                times_list.append(time.time() - start_time)
                scores_list.append(score)
                print("Iteration n°", iteration + 1, "done. Score:", score)
            best_scores[penalty_id, algorithm_id] = min(scores_list)
            mean_scores[penalty_id, algorithm_id] = np.mean(scores_list)
            scores_std[penalty_id, algorithm_id] = np.std(scores_list)
            average_execution_times[penalty_id, algorithm_id] = np.mean(times_list)

        # Running random walk, by giving it the maximum time needed by any algorithm run
        total_execution_time = 10
        average_execution_times[penalty_id, -1] = total_execution_time
        scores_list = list()
        print("\nRandom Walk")
        for iteration in range(n_iterations):
            start_time = time.time()
            best_score = np.inf
            while (time.time() - start_time) < total_execution_time:
                sequence, actions, score = random_walk(environment)
                if score < best_score:
                    best_score = score
            scores_list.append(best_score)
            print("Iteration n°", iteration + 1, "done. Score:", score)
        best_scores[penalty_id, -1] = min(scores_list)
        mean_scores[penalty_id, -1] = np.mean(scores_list)
        scores_std[penalty_id, -1] = np.std(scores_list)

    indices = penalty_factor_list
    columns = list(algorithms.keys()) + ["Random Walk"]
    writer = pd.ExcelWriter(
        "Aggregated_scores_by_penalty_factor.xlsx", engine="xlsxwriter"
    )

    mean_dataframe = pd.DataFrame(
        data=mean_scores, columns=columns, index=penalty_factor_list
    )
    std_dataframe = pd.DataFrame(
        data=scores_std, columns=columns, index=penalty_factor_list
    )
    min_dataframe = pd.DataFrame(
        data=best_scores, columns=columns, index=penalty_factor_list
    )
    time_dataframe = pd.DataFrame(
        data=average_execution_times, columns=columns, index=penalty_factor_list
    )

    mean_dataframe.to_excel(writer, sheet_name="Mean score")
    std_dataframe.to_excel(writer, sheet_name="Standard deviation of score")
    min_dataframe.to_excel(writer, sheet_name="Min score")
    time_dataframe.to_excel(writer, sheet_name="Average time")

    writer.close()
