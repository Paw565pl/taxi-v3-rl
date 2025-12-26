import numpy as np
import gymnasium as gym
from gymnasium import spaces


class OneHotObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Discrete), (
            "requires env with discrete observation space"
        )

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(env.observation_space.n,), dtype=np.float32
        )
        self.num_categories = env.observation_space.n

    def observation(self, observation):
        one_hot = np.zeros(self.num_categories, dtype=np.float32)
        one_hot[int(observation)] = 1

        return one_hot
