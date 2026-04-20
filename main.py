"""
Setup a CSTR environment with a setpoint change (see https://maximilianb2.github.io/pc-gym/#welcome)
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import pcgym
import matplotlib.pyplot as plt

from classes.environment import EnvironmentWrapper
from solvers.cnmcts import cnmcts
from solvers.crbnmcts import crbnmcts
from solvers.cnrpa import run_cnrpa
from solvers.baselines import get_and_train_baseline, run_baseline
from utils.models import *

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":

    # Functioning problems list

    problems = {
        # # CSTR problem
        "cstr": {
            "inputs": {
                "problem": "cstr",
                "initial_state": np.array([0.8, 330, 0.8]),
                "n_steps": 100,
                "T": 25,
                "penalty_factor": np.zeros((1, 1)),
            },
            "plotting_indices": [0],
            "setpoints_names": ["C_A"],
        },
        # Example for Nonsmooth control
        "nonsmooth_control": {
            "inputs": {
                "problem": "nonsmooth_control",
                "initial_state": np.array([0, 0, 0.2]),
                "penalty_factor": np.zeros((1, 1)),
                "n_steps": 100,
                "T": 25,
            },
            "plotting_indices": [0],
            "setpoints_names": ["X_1"],
        },
        # Example for Multistage Extraction Column
        # "multistage_extraction": {
        #     "inputs": {
        #         "problem": "multistage_extraction",
        #         "initial_state": np.array(
        #             [0.55, 0.3, 0.45, 0.25, 0.4, 0.20, 0.35, 0.15, 0.25, 0.1, 0.3]
        #         ),
        #         "penalty_factor": np.zeros((2, 2)),
        #         "n_steps": 500,
        #         "T": 10,
        #     },
        #     "plotting_indices": [0],
        #     "setpoints_names": ["X_1"],
        # }
    }

    # Defining setpoints

    ## CSTR
    problems["cstr"]["inputs"]["set_points"] = {
        "Ca": [0.85 for i in range(int(problems["cstr"]["inputs"]["n_steps"] / 2))]
        + [0.9 for i in range(int(problems["cstr"]["inputs"]["n_steps"] / 2))],
    }
    problems["cstr"]["set_plots"] = [
        problems["cstr"]["inputs"]["set_points"]["Ca"],
    ]

    ## Nonsmooth Control
    problems["nonsmooth_control"]["inputs"]["set_points"] = {
        "X1": [
            0.3
            for i in range(int(problems["nonsmooth_control"]["inputs"]["n_steps"] / 2))
        ]
        + [
            0.4
            for i in range(int(problems["nonsmooth_control"]["inputs"]["n_steps"] / 2))
        ],
    }
    problems["nonsmooth_control"]["set_plots"] = [
        problems["nonsmooth_control"]["inputs"]["set_points"]["X1"],
    ]

    # ## Multistage Extraction
    # problems["multistage_extraction"]["inputs"]["set_points"] = {
    #     "X1": [0.2 for i in range(int(problems["multistage_extraction"]["inputs"]["n_steps"] / 2))]
    #     + [0.4 for i in range(int(problems["multistage_extraction"]["inputs"]["n_steps"] / 2))],
    # }
    # problems["multistage_extraction"]["set_plots"] = [
    #     problems["multistage_extraction"]["inputs"]["set_points"]["X1"],
    # ]

    # Iterating through different parameters
    nesting_levels = {
        1: {"bandwidth": 25, "n_policies": 500, "half_life_divider": 1},
        2: {"bandwidth": 20, "n_policies": 100, "half_life_divider": 5},
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
        }
        algorithms["cRbNMCTS level " + str(level)] = {
            "function": crbnmcts,
            "parameters": {
                "level": level,
                "bandwidth": nesting_levels[level]["bandwidth"],
            },
        }
        algorithms["Gaussian cNRPA level " + str(level)] = {
            "function": run_cnrpa,
            "parameters": {
                "level": level,
                "n_policies": nesting_levels[level]["n_policies"],
                "policy_type": "gaussian",
                "half_life_divider": nesting_levels[level]["half_life_divider"],
            },
        }
        algorithms["By Region cNRPA level " + str(level)] = {
            "function": run_cnrpa,
            "parameters": {
                "level": level,
                "n_policies": nesting_levels[level]["n_policies"],
                "policy_type": "by_region",
                "half_life_divider": nesting_levels[level]["half_life_divider"],
            },
        }
    baselines = ["PPO", "A2C", "DDPG", "SAC"]
    baselines_parameters = {
        "n_steps_learning": 100000,
        "learning_rate": 0.01,
    }

    # Number of iterations per algorithm
    n_iterations = 10

    # Storing values
    best_scores = np.zeros((len(problems), len(algorithms) + 1))
    mean_scores = np.zeros((len(problems), len(algorithms) + 1))
    scores_std = np.zeros((len(problems), len(algorithms) + 1))
    average_execution_times = np.zeros((len(problems), len(algorithms) + 1))
    figures_directory = "./figures"

    for problem_id, problem in enumerate(problems.keys()):
        print("\n\nCurrent problem:", problem)
        environment, observation_space = get_environment(**problems[problem]["inputs"])
        environment = EnvironmentWrapper(
            environment,
            problems[problem]["inputs"]["n_steps"],
            problems[problem]["inputs"]["penalty_factor"],
        )
        sequences_to_plot = dict()

        # Iterating over implemented algorithms
        for algorithm_id, algorithm in enumerate(algorithms.keys()):
            print("\nRunning ", algorithm)
            # Local storage
            scores_list = list()
            times_list = list()
            sequences_list = list()

            # Running n_iterations of the same algorithm
            for iteration in range(n_iterations):
                environment.reset()
                start_time = time.time()
                sequence, actions, score = algorithms[algorithm]["function"](
                    environment, **algorithms[algorithm]["parameters"]
                )
                times_list.append(time.time() - start_time)
                scores_list.append(score)
                sequences_list.append(sequence)
                print(
                    "Iteration n°",
                    iteration + 1,
                    "done. Score:",
                    "{:.4f}".format(score),
                )
            best_index = np.argmin(scores_list)
            best_scores[problem_id, algorithm_id] = scores_list[best_index]
            sequences_to_plot[algorithm] = sequences_list[best_index]
            mean_scores[problem_id, algorithm_id] = np.mean(scores_list)
            scores_std[problem_id, algorithm_id] = np.std(scores_list)
            average_execution_times[problem_id, algorithm_id] = np.mean(times_list)

        # Finding the best performing baseline
        best_score = np.inf
        best_sequence = None
        best_baseline = None
        corresponding_time = None
        for baseline in baselines:
            # Training
            print("\n" + baseline)
            start_time = time.time()
            environment.reset()
            model = get_and_train_baseline(
                environment, baseline, **baselines_parameters
            )
            # Evaluating
            environment.reset()
            sequence, actions, score = run_baseline(environment, model)
            execution_time = time.time() - start_time
            if score < best_score:
                best_score = score
                best_sequence = sequence
                best_baseline = baseline
                corresponding_time = execution_time

        sequences_to_plot[best_baseline] = best_sequence

        # Adapting values in the sequences by denormalizing them
        for algorithm, sequence in sequences_to_plot.items():
            sequences_to_plot[algorithm] = [
                element * (observation_space["high"] - observation_space["low"]) / 2
                + (observation_space["high"] + observation_space["low"]) / 2
                for element in sequence
            ]

        # Plotting and saving figure
        figure_name = figures_directory + "/" + problem + "_comparison.jpeg"
        figure_title = ""
        for algorithm_id, algorithm in enumerate(algorithms.keys()):
            figure_title += (
                algorithm
                + ": "
                + "{:.4f}".format(best_scores[problem_id, algorithm_id])
                + ". "
            )
        figure_title += best_baseline + ": " + "{:.4f}".format(best_score)
        n_graphs = len(problems[problem]["setpoints_names"])
        figure, axes = plt.subplots(n_graphs, layout="constrained", figsize=(15, 10))
        if n_graphs > 1:
            for i in range(n_graphs):
                # Setpoints
                axes[i].plot(
                    problems[problem]["set_plots"][i],
                    color="k",
                    linestyle="--",
                    label="Setpoints",
                )
                for algorithm, sequence in sequences_to_plot.items():
                    axes[i].plot(
                        [
                            element[problems[problem]["plotting_indices"][i]]
                            for element in sequence
                        ],
                        label=algorithm,
                    )
                axes[i].legend()
                axes[i].set_title("$" + problems[problem]["setpoints_names"][i] + "$")

        else:
            axes.plot(
                problems[problem]["set_plots"][0],
                color="k",
                linestyle="--",
                label="Setpoints",
            )
            for algorithm, sequence in sequences_to_plot.items():
                axes.plot(
                    [
                        element[problems[problem]["plotting_indices"][0]]
                        for element in sequence
                    ],
                    label=algorithm,
                )
            axes.legend()
            axes.set_title("$" + problems[problem]["setpoints_names"][0] + "$")
        figure.suptitle(figure_title)
        figure.savefig(figure_name)
        plt.close()

    indices = list(problems.keys())
    columns = list(algorithms.keys()) + ["Baseline"]
    writer = pd.ExcelWriter("Aggregated_scores_by_problem.xlsx", engine="xlsxwriter")

    mean_dataframe = pd.DataFrame(data=mean_scores, columns=columns, index=indices)
    std_dataframe = pd.DataFrame(data=scores_std, columns=columns, index=indices)
    min_dataframe = pd.DataFrame(data=best_scores, columns=columns, index=indices)
    time_dataframe = pd.DataFrame(
        data=average_execution_times, columns=columns, index=indices
    )

    mean_dataframe.to_excel(writer, sheet_name="Mean score")
    std_dataframe.to_excel(writer, sheet_name="Standard deviation of score")
    min_dataframe.to_excel(writer, sheet_name="Min score")
    time_dataframe.to_excel(writer, sheet_name="Average time")

    writer.close()
