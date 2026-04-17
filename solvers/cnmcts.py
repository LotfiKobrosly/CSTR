from copy import deepcopy
import numpy as np

from classes.environment import EnvironmentWrapper
from utils.models import code


def cnmcts(environment: EnvironmentWrapper, level: int = 1, bandwidth: int = 20):
    if level == 0:
        while not environment.is_final():
            action = environment.sample_random_action()
            environment.step(action)
        return environment.sequence, environment.actions, environment.score

    else:
        while not environment.is_final():
            actions_list = list()
            while len(actions_list) < bandwidth:
                action = environment.sample_random_action()
                if not np.any([np.all(action == element) for element in actions_list]):
                    actions_list.append(action)
            for action in actions_list:
                temporary_environment = deepcopy(environment)
                temporary_environment.step(action)
                sequence, actions, score = cnmcts(
                    temporary_environment, level - 1, bandwidth
                )
                if np.absolute(score) < np.absolute(environment.best_score):
                    environment.best_sequence = sequence[:]
                    environment.best_actions = actions[:]
                    environment.best_score = score
            environment.step(environment.best_actions[environment.current_timestamp])

        return (
            environment.best_sequence,
            environment.best_actions,
            environment.best_score,
        )
