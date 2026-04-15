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
        self.truncate = False

    def is_final(self):
        return self.done or self.current_timestamp >= self.horizon

    def allow_truncation(self):
        self.truncate = True

    def disallow_truncation(self):
        self.truncate = False

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
        self.instant_reward = 0

    def truncate_observation(self, observation):
        for component_index, component in enumerate(observation):
            if component > 2:
                observation[component_index] = 1
            if component < -2:
                observation[component_index] = -1
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.environment.step(action)
        if self.truncate:
            observation = self.truncate_observation(observation)
            self.environment.obs = observation[:]
        observation = code(observation)
        if not isinstance(action, np.ndarray):
            action = np.array([action])
        action = action.reshape(1, -1)
        if self.actions:
            penalty = np.absolute(action - self.actions[-1])
        else:
            penalty = np.zeros((max(action.shape), max(action.shape)))

        self.actions.append(action)
        self.current_timestamp += 1
        self.done = terminated
        self.sequence.append(observation)
        self.current_state = observation
        self.instant_reward = np.absolute(reward)
        self.score += self.instant_reward + (penalty @ self.penalty_factor @ penalty.T).flatten()[0]
        self.cumulative_distance_to_true_value += np.absolute(reward)

    def sample_random_action(self):
        return self.environment.action_space.sample()

    def get_last_reward(self):
        return self.instant_reward
