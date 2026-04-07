from copy import deepcopy
import numpy as np

from environment import EnvironmentWrapper


def cnmcts(environment: EnvironmentWrapper, level: int = 1, bandwidth: int = 20):
    if level == 0:
        while not environment.is_final():
            if (
                environment.code(environment.current_state)
                in environment.states_actions.keys()
            ):
                choices = np.arange(
                    len(
                        environment.states_actions[
                            environment.code(environment.current_state)
                        ]
                    )
                )
                chosen_index = np.random.choice(choices)
                action = environment.states_actions[
                    environment.code(environment.current_state)
                ][chosen_index]
            else:
                action = environment.sample_random_action()
            environment.step(action)
        return environment.sequence, environment.actions, environment.score

    else:
        while not environment.is_final():
            if (
                not environment.code(environment.current_state)
                in environment.states_actions.keys()
            ):
                environment.states_actions[
                    environment.code(environment.current_state)
                ] = [environment.sample_random_action() for _ in range(bandwidth)]
            for action in environment.states_actions[
                environment.code(environment.current_state)
            ]:
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
