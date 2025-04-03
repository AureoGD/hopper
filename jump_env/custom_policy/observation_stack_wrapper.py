import gymnasium as gym
import numpy as np
from collections import deque

class ObservationStackWrapper(gym.ObservationWrapper):
    def __init__(self, env, stack_size=5):
        super(ObservationStackWrapper, self).__init__(env)
        self.stack_size = stack_size
        self.observation_space = gym.spaces.Box(
            low=np.repeat(env.observation_space.low[None, :], stack_size, axis=0),
            high=np.repeat(env.observation_space.high[None, :], stack_size, axis=0),
            dtype=np.float32
        )
        self.stacked_obs = deque(maxlen=stack_size)

    def observation(self, obs):
        self.stacked_obs.append(obs)
        while len(self.stacked_obs) < self.stack_size:
            self.stacked_obs.append(obs)
        return np.array(self.stacked_obs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.stacked_obs.clear()
        for _ in range(self.stack_size):
            self.stacked_obs.append(obs)
        return self.observation(obs), info
