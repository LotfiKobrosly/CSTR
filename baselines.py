import numpy as np 
from stable_baselines3 import PPO
from environment import EnvironmentWrapper

def get_and_train_ppo(environment: EnvironmentWrapper, n_steps_learning: int = int(3e4), learning_rate: float=0.001):
    return PPO('MlpPolicy', environment.environment, verbose=1, learning_rate=learning_rate).learn(n_steps_learning)

def run_ppo(environment: EnvironmentWrapper, ppo_policy: dict()):
    while environment.current_timestamp <= environment.horizon:
        action, observation = ppo_policy.predict(environment.current_state)
        print(action)
        #action = np.array([action]).reshape(-1)
        environment.step(action)
    return environment.sequence, environment.actions, environment.score

