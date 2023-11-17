from metroid_env import MetroidGymEnv

from stable_baselines3 import PPO

import configs as c

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
        env = MetroidGymEnv(config)
        env.reset(seed=(seed + rank))
        return env
    
    set_random_seed(seed)
    return _init

if __name__ == "__main__":

    env = make_env(0, c.replay)()

    file_name = 'sessions/session_2023-11-17_00:39:01.533645/mai_655360_steps'
    model = PPO.load(file_name, env=env)

    obs, info = env.reset()
    for i in range(1000):

        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()

    env.close()