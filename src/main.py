from random import randint
from pathlib import Path
import uuid

from metroid_env import MetroidGymEnv

from stable_baselines3 import PPO

import basic_config as bc

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList


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


if __name__ == '__main__':

    n_steps = bc.config["max_steps"]
    n_envs = 4

    session_id = str(uuid.uuid4())[:8]
    session_path = Path(f'sessions/session_{session_id}')
    
    # create environment
    env = SubprocVecEnv([make_env(i, bc.config) for i in range(n_envs)])
    # check_env(env)

    # establish callbacks
    enable_callbacks = False
    callbacks = []

    if enable_callbacks:
        checkpoint_callback = CheckpointCallback(save_freq=n_steps, 
                                                save_path=session_path, 
                                                name_prefix='mai')
        callbacks.append(checkpoint_callback)

    callbacks = CallbackList(callbacks)

    # rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    model = PPO('CnnPolicy', env, verbose=1, batch_size=n_steps, n_steps=n_steps*n_envs)

    learning_iters = 2
    for _ in range(learning_iters):
        model.learn(total_timesteps=n_steps, callback=callbacks)

    # close environments
    env.close()
