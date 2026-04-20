"""
This file implements a Continuous Reward-based Nested Monte Carlo Tree Search
"""

from copy import deepcopy
import numpy as np

from classes.environment import EnvironmentWrapper
from utils.models import code


def crbnmcts(
    environment: EnvironmentWrapper,
    level: int = 1,
    bandwidth: int = 20,
    action: np.ndarray = None,
):
    if level == 0:
        environment.step(action)
        return (
            environment.sequence,
            environment.actions,
            environment.score,
        )

    else:
        if not (action is None):
            environment.step(action)
        while not environment.is_final():
            actions_list = list()
            best_action = None
            best_score = environment.best_score
            while len(actions_list) < bandwidth:
                new_action = environment.sample_random_action()
                if not np.any(
                    [np.all(new_action == element) for element in actions_list]
                ):
                    actions_list.append(new_action)
            temporary_environments_list = [
                deepcopy(environment) for _ in range(bandwidth)
            ]
            for new_action, temporary_environment in zip(
                actions_list, temporary_environments_list
            ):
                sequence, actions, score = crbnmcts(
                    temporary_environment,
                    level=level - 1,
                    bandwidth=bandwidth,
                    action=new_action,
                )
                if score < best_score:
                    best_action = new_action
                    best_score = score
                if temporary_environment.is_final() and (
                    score < environment.best_score
                ):
                    environment.best_score = score
                    environment.best_sequence = sequence[:]
                    environment.best_actions = actions[:]
            if best_action is None:
                best_action = environment.best_actions[environment.current_timestamp]
            environment.step(best_action)
        if environment.score < environment.best_score:
            environment.best_score = environment.score
            environment.best_sequence = environment.sequence[:]
            environment.best_actions = environment.actions[:]
        return (
            environment.best_sequence,
            environment.best_actions,
            environment.best_score,
        )
