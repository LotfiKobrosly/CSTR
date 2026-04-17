"""
This file implements a Continuous Reward-based Nested Monte Carlo Tree Search
"""

from copy import deepcopy
import numpy as np

from classes.environment import EnvironmentWrapper
from utils.models import code


def crbnmcts(
    environment: EnvironmentWrapper, level: int = 1, bandwidth: int = 20, action=None
):
    if level == 0:
        environment.step(action)
        return environment.sequence, environment.get_last_reward()

    else:
        if not (action is None):
            environment.step(action)
        while not environment.is_final():
            actions_list = list()
            best_action = None
            best_reward = np.inf
            while len(actions_list) < bandwidth:
                new_action = environment.sample_random_action()
                if not np.any(
                    [np.all(new_action == element) for element in actions_list]
                ):
                    actions_list.append(new_action)
            for new_action in actions_list:
                temporary_environment = deepcopy(environment)
                sequence, reward = crbnmcts(
                    temporary_environment, level - 1, bandwidth, new_action
                )
                if reward < best_reward:
                    # environment.best_sequence = sequence[:]
                    best_action = new_action
                    best_reward = reward
            environment.step(best_action)

        return (
            environment.sequence,
            environment.score,
        )
