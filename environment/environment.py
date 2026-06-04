import gymnasium as gym
import numpy as np
import ale_py
from environment.wrappers import make_env

class AtariEnvironment:
    def __init__(self, env_id='ALE/Pong-v5', render_mode=None, seed=None):
        self.env_id = env_id
        self.render_mode = render_mode
        self.seed = seed
        
        gym.register_envs(ale_py)
        self.env = make_env(env_id, render_mode)
        
        if seed is not None:
            self.env.reset(seed=seed)
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.n_actions = self.env.action_space.n

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()

    def sample_action(self):
        return self.env.action_space.sample()

def make_atari_env(env_id='ALE/Pong-v5', render_mode=None, seed=None):
    return AtariEnvironment(env_id, render_mode, seed)