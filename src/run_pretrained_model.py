from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

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

    file_name = 'sessions/session_4bd29/mai_32768000_steps'
    model = PPO.load(file_name, env=env)

    obs, info = env.reset()
    while True:

        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()

    env.close()