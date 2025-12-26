from typing import Literal

import gymnasium as gym
from gymnasium import Env
from one_hot_observation import OneHotObservation
from stable_baselines3.common.monitor import Monitor


def make_env(render_mode: Literal["human", "rgb_array"] | None = None) -> Env:
    env = gym.make("Taxi-v3", render_mode=render_mode)

    env = Monitor(env)
    env = OneHotObservation(env)

    return env
