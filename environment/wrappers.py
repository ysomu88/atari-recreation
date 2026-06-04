import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import cv2


class NoopResetWrapper(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        self._obs_buffer = deque(maxlen=2)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        # Safe max — if only one frame in buffer, use it twice
        if len(self._obs_buffer) == 1:
            max_frame = self._obs_buffer[0]
        else:
            max_frame = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        return max_frame, total_reward, terminated, truncated, info


class WarpFrameWrapper(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(height, width, 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis].astype(np.uint8)


class FrameStackWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frame_buffer = deque(maxlen=n_frames)
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(n_frames, obs_shape[0], obs_shape[1]),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frame_buffer.append(np.zeros(
                (self.observation_space.shape[1], self.observation_space.shape[2]),
                dtype=np.uint8
            ))
        stacked = self.observation(obs)
        return stacked, info

    def observation(self, obs):
        frame = obs[:, :, 0] if obs.ndim == 3 else obs
        self.frame_buffer.append(frame)
        return np.stack(list(self.frame_buffer), axis=0).astype(np.uint8)


def make_env(env_id='ALE/Pong-v5', render_mode=None):
    import ale_py
    gym.register_envs(ale_py)
    env = gym.make(env_id, render_mode=render_mode)
    env = NoopResetWrapper(env)
    env = MaxAndSkipWrapper(env)
    env = WarpFrameWrapper(env)
    env = FrameStackWrapper(env)
    return env