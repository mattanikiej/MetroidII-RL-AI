from random import randint

from metroid_env import MetroidGymEnv

from pyboy import PyBoy

from stable_baselines3 import PPO

import basic_config as bc
import memory_constants as c

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

    code from https://github.com/PWhiddy/PokemonRedExperiments/blob/master/baselines/run_baseline_parallel_fast.py
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
    return _init()

# pyboy = PyBoy('../ROMs/Metroid2.gb')
# f = open("../states/post_start_screen.state", "rb")
# pyboy.load_state(f)
# f.close()

if __name__ == '__main__':
    n_steps = bc.config["max_steps"]
    n_envs = 2
    envs = []
    config = bc.config
    for i in range(n_envs):
        env = make_env(i, config)
        envs.append(env)
    
    env = SubprocVecEnv(envs)
    # env = MetroidGymEnv(bc.config)
    # check_env(env)

    # rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    # model = PPO('CnnPolicy', env, verbose=1, batch_size=n_steps, n_steps=n_steps*n_envs)

    # learning_iters = 2
    # for _ in range(learning_iters):
    #     model.learn(total_timesteps=n_steps*n_envs*10)

    # env.close()

    # i = 1
    # while True:
    #     pyboy.tick()

    #     if i % 1000 == 0:
    #         if input("Save state? ") == 'y':
    #             name = input("state name: ")
    #             with open("../states/"+name, 'wb') as f:
    #                 pyboy.save_state(f)
        
    #     i += 1