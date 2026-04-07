"""
Setup a CSTR environment with a setpoint change (see https://maximilianb2.github.io/pc-gym/#welcome)
"""

import time
import numpy as np
import pcgym
import matplotlib.pyplot as plt

from cnmcts import cnmcts
from cnrpa import run_cnrpa
from models import *
from environment import EnvironmentWrapper


if __name__ == "__main__":

    # Global inputs
    T = 25
    n_steps = 50
    penalty_factor = 5

    # CSTR problem
    problem = "cstr"
    initial_state = np.array([0.8, 330, 0.8])

    # Resulting inputs
    inputs = {
        "T": T,
        "n_steps": n_steps,
        "initial_state": initial_state,
        "problem": problem,
    }

    execution_time = 0

    # cNMCTS
    ## Environment
    environment, observation_space = get_environment(**inputs)
    environment = EnvironmentWrapper(environment, n_steps, penalty_factor)
    print("Environment steup for cNMCTS")
    start_time = time.time()
    ## Running cNMCTS
    sequence_cnmcts, actions_cnmcts, score_cnmcts = cnmcts(
        environment,
        level=2,
        bandwidth=5,
    )
    total_time = time.time() - start_time
    if total_time > execution_time:
        execution_time = total_time
    sequence_cnmcts = [
        element * (observation_space["high"] - observation_space["low"]) / 2
        + (observation_space["high"] + observation_space["low"]) / 2
        for element in sequence_cnmcts
    ]

    print("Final score for cNMCTS:", "{:.3f}".format(score_cnmcts))
    print("Total time for cNMCTS:", "{:.3f}".format(total_time))

    # cNRPA with gaussian policy
    ## Environment
    environment, observation_space = get_environment(**inputs)
    environment = EnvironmentWrapper(environment, n_steps, penalty_factor)
    print("Environment steup for gaussian cNRPA")
    ## Runnincg cNRPA
    start_time = time.time()
    sequence_cnrpa_gaussian, actions_cnrpa_gaussian, score_cnrpa_gaussian = run_cnrpa(
        environment,
        level=2,
        n_policies=25,
        policy_type="gaussian",
        observation_space=observation_space,
    )

    total_time = time.time() - start_time
    if total_time > execution_time:
        execution_time = total_time
    sequence_cnrpa_gaussian = [
        element * (observation_space["high"] - observation_space["low"]) / 2
        + (observation_space["high"] + observation_space["low"]) / 2
        for element in sequence_cnrpa_gaussian
    ]

    print("Final score for cNRPA gaussian:", "{:.3f}".format(score_cnrpa_gaussian))
    print("Total time for cNRPA gaussian:", "{:.3f}".format(total_time))


    # cNRPA with by region policy
    ## Environment
    environment, observation_space = get_environment(**inputs)
    environment = EnvironmentWrapper(environment, n_steps, penalty_factor)
    print("Environment steup for by region cNRPA")
    ## Runnincg cNRPA
    start_time = time.time()
    sequence_cnrpa_by_region, actions_cnrpa_by_region, score_cnrpa_by_region = run_cnrpa(
        environment,
        level=2,
        n_policies=25,
        policy_type="by_region",
        observation_space=observation_space,
    )

    total_time = time.time() - start_time
    if total_time > execution_time:
        execution_time = total_time
    sequence_cnrpa_by_region = [
        element * (observation_space["high"] - observation_space["low"]) / 2
        + (observation_space["high"] + observation_space["low"]) / 2
        for element in sequence_cnrpa_by_region
    ]

    print("Final score for cNRPA by region:", "{:.3f}".format(score_cnrpa_by_region))
    print("Total time for cNRPA by region:", "{:.3f}".format(total_time))

    # Random Walk
    ## Environment
    environment, observation_space = get_environment(**inputs)
    environment = EnvironmentWrapper(environment, n_steps, penalty_factor)
    print("Environment steup for Random Walk")
    # Running Random Walk
    start_time = time.time()
    random_walk_score = np.inf
    while time.time() - start_time < execution_time:
        done = False
        environment.reset()
        while not environment.is_final():
            environment.step(environment.sample_random_action())
        score = environment.score
        if score < random_walk_score:
            random_walk_score = score
            random_walk_sequence = environment.sequence
            random_walk_actions = environment.actions
    total_time = time.time() - start_time
    print("Final score for Randsom Walk:", "{:.3f}".format(random_walk_score))
    print("Total time for Random Walk:", "{:.3f}".format(total_time))

    figure, axes = plt.subplots(4, 3, layout="constrained")
    sequences = [sequence_cnmcts, sequence_cnrpa_gaussian, sequence_cnrpa_by_region, random_walk_sequence]
    for counter, algorithm in enumerate(["cNMCTS", "Gaussian_cNRPA", "By_Region_cNRPA", "Random Walk"]):
        current_axe = axes[counter]
        plot_1, plot_2, plot_3 = list(), list(), list()
        for element in sequences[counter]:
            plot_1.append(element[0])
            plot_2.append(element[1])
            plot_3.append(element[2])
        current_axe[0].plot(plot_1)
        current_axe[1].plot(plot_2)
        current_axe[2].plot(plot_3)
        current_axe[0].set_title("$C_A$ of " + algorithm)
        current_axe[1].set_title("T of " + algorithm)
        current_axe[2].set_title("Setpoints of $C_A$")
    plt.show()
