import numpy as np

from constants import RANDOM_SEED
from models import code


class EnvironmentWrapper:
    def __init__(self, environment, simulation_horizon: float, penalty_factor: np.ndarray):
        self.environment = environment
        # Initialize the environment
        observation, info = environment.reset(RANDOM_SEED)
        self.current_state = code(observation)
        self.sequence = [self.current_state]
        self.best_sequence = [self.current_state]
        self.actions = list()
        self.best_actions = list()
        self.states_actions = dict()
        self.current_timestamp = 0
        self.horizon = simulation_horizon
        self.best_score = np.inf
        self.done = False
        self.score = 0
        self.cumulative_distance_to_true_value = 0
        self.penalty_factor = penalty_factor

    def is_final(self):
        return self.done or self.current_timestamp >= self.horizon

    def reset(self):
        observation, info = self.environment.reset()
        self.current_state = code(observation)
        self.sequence = [self.current_state]
        self.best_sequence = [self.current_state]
        self.actions = list()
        self.best_actions = list()
        self.states_actions = dict()
        self.current_timestamp = 0
        self.best_score = np.inf
        self.done = False
        self.score = 0
        self.cumulative_distance_to_true_value = 0

    def truncate_observation(self, observation):
        for component_index, component in enumerate(observation):
            if component > 1:
                observation[component_index] = 1
            if component < -1:
                observation[component_index] = -1
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.environment.step(action)
        #observation = self.truncate_observation(observation)
        observation = code(observation)
        if self.actions:
            penalty = np.absolute(action - self.actions[-1])
        else:
            penalty = np.array([0])
        if not isinstance(action, np.ndarray):
            action = np.array([action])
        action = action.reshape(1, -1)

        self.actions.append(action)
        self.current_timestamp += 1
        self.done = terminated
        self.sequence.append(observation)
        self.current_state = observation
        self.score += np.absolute(reward) + (penalty @ self.penalty_factor @ penalty.T).flatten()[0]
        self.cumulative_distance_to_true_value += np.absolute(reward)

    def sample_random_action(self):
        return self.environment.action_space.sample()
