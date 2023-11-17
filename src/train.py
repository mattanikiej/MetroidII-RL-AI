from pathlib import Path
from uuid import uuid4

from metroid_env import MetroidGymEnv

from stable_baselines3 import PPO

import configs as c

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList


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

    cfg = c.short
    n_steps = cfg["max_steps"]
    n_envs = 6

    session_id = str(uuid4())[:5]
    session_path = Path(f'sessions/session_{session_id}')
    tb_path = Path(f'sessions/session_{session_id}/tb')

    # create environment
    env = SubprocVecEnv([make_env(i, cfg) for i in range(n_envs)])

    # establish callbacks
    enable_callbacks = True
    callbacks = []

    if enable_callbacks:
        checkpoint_callback = CheckpointCallback(save_freq=n_steps, 
                                                 save_path=session_path, 
                                                 name_prefix='mai')

        callbacks.append(checkpoint_callback)

    callbacks = CallbackList(callbacks)

    n_epochs = 10
    model = PPO('CnnPolicy', env, verbose=1, n_epochs=n_epochs, batch_size=128, n_steps=n_steps // 8, tensorboard_log=tb_path)

    model.learn(total_timesteps=n_steps*n_envs*100, callback=callbacks)

    # close environments
    env.close()
