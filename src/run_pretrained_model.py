from stable_baselines3 import PPO, DQN
from stable_baselines3.common.utils import set_random_seed

import torch

from metroid_env import MetroidGymEnv
import configs as c


def make_env(rank, config, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = MetroidGymEnv(config)
        env.reset(seed=(seed + rank))
        return env
    
    set_random_seed(seed)
    return _init

if __name__ == "__main__":

    env = make_env(0, c.replay)()

    file_name = 'sessions/session_24636/mai_25231360_steps.zip'
    model = DQN.load(file_name, env=env)
    model.verbose = 1
    model.batch_size = 256


    obs, info = env.reset()
    while True:

        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        obs = env.render()

    env.close()