import numpy as np 
from stable_baselines3 import PPO
from pcgym import oracle
from environment import EnvironmentWrapper

def get_and_train_ppo(environment: EnvironmentWrapper, n_steps_learning: int = int(3e4), learning_rate: float=0.001):
    return PPO('MlpPolicy', environment.environment, verbose=1, learning_rate=learning_rate).learn(n_steps_learning)
