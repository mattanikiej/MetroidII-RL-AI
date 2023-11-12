from random import randint

from metroid_env import MetroidGymEnv

from stable_baselines3 import PPO

import basic_config as bc

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed


def make_env(rank, config, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        # choose random start state
        state_path = '../states/'
        states = [
            'bottom_of_pit.state', 
            'inside_pit.state', 
            'past_first_door.state', 
            'post_start_screen.state'
            ]
        
        i = randint(0, len(states)-1)
        state = states[i]

        config['initial_state'] = state_path + state

        env = MetroidGymEnv(config)
        env.reset(seed=(seed + rank))

        return env
    
    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    n_steps = bc.config["max_steps"]
    n_envs = 1
    
    env = SubprocVecEnv([make_env(i, bc.config) for i in range(n_envs)])
    # env = MetroidGymEnv(bc.config)
    # check_env(env)

    # rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    model = PPO('CnnPolicy', env, verbose=1, batch_size=n_steps, n_steps=n_steps*n_envs)

    learning_iters = 2
    for _ in range(learning_iters):
        model.learn(total_timesteps=n_steps*n_envs)

    # close environments
    env.close()
