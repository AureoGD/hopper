import numpy as np
from gymnasium import ObservationWrapper


class ObservationStackWrapper(ObservationWrapper):
    def __init__(self, env, stack_size=10):
        super().__init__(env)
        self.stack_size = stack_size
        self.history = None

        obs_shape = env.observation_space.shape
        self.observation_space = env.observation_space.__class__(
            shape=(stack_size,) + obs_shape,
            low=np.repeat(env.observation_space.low[None, :], stack_size, axis=0),
            high=np.repeat(env.observation_space.high[None, :], stack_size, axis=0),
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.asarray(obs)

        self.history = [obs for _ in range(self.stack_size)]
        return self._get_obs(), info

    def observation(self, obs):
        obs = np.asarray(obs)
        self.history.pop(0)
        self.history.append(obs)
        return self._get_obs()

    def _get_obs(self):
        return np.stack(self.history, axis=0)
