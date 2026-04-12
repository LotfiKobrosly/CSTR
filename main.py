"""
Setup a CSTR environment with a setpoint change (see https://maximilianb2.github.io/pc-gym/#welcome)
"""

import time
import numpy as np
import pandas as pd
import pcgym
import matplotlib.pyplot as plt

from cnmcts import cnmcts
from cnrpa import run_cnrpa
from random_walk import random_walk
from models import *
from environment import EnvironmentWrapper


if __name__ == "__main__":

    # Running algorithms on the CSTR problem
    # Global inputs
    T = 25
    n_steps = 50

    # CSTR problem
    problem = "cstr"
    initial_state = np.array([0.8, 330, 0.8])
    set_points = {
        "Ca": [0.85 for i in range(int(n_steps / 2))]
        + [0.9 for i in range(int(n_steps / 2))]
    }
    """
    # Multistage extraction column problem
    problem = "multistage_extraction"
    initial_state = np.array([0.8, 330, 0.8])
    set_points = {
        "Ca": [0.85 for i in range(int(n_steps / 2))]
        + [0.9 for i in range(int(n_steps / 2))]
    }

    # Nonsmooth control problem
    problem = "nonsmooth_control"
    initial_state = np.array([0.8, 330, 0.8])
    set_points = {
        "Ca": [0.85 for i in range(int(n_steps / 2))]
        + [0.9 for i in range(int(n_steps / 2))]
    }

    # Photo production problem
    problem = "photo_production"
    initial_state = np.array([0.1, 20.0, 0.01])
    set_points = dict()
    """

    # Inputs
    inputs = {
        "T": T,
        "n_steps": n_steps,
        "initial_state": initial_state,
        "problem": problem,
        "set_points": set_points,
    }

    # Iterating through different parameters
    penalty_factor_list = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    nesting_levels = {
        1: {"bandwidth": 25, "n_policies": 500},
        2: {"bandwidth": 5, "n_policies": 20},
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
            },
            "plots": [list(), list()],
        }
        algorithms["By Region cNRPA level " + str(level)] = {
            "function": run_cnrpa,
            "parameters": {
                "level": level,
                "n_policies": nesting_levels[level]["n_policies"],
                "policy_type": "by_region",
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
