from pathlib import Path
from uuid import uuid4

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, vec_transpose
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

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


if __name__ == '__main__':

    cfg = c.basic
    n_steps = cfg["max_steps"]
    n_envs = cfg["n_envs"]

    session_id = str(uuid4())[:5]
    session_path = Path(f'sessions/session_{session_id}')
    tb_path = Path(f'sessions/session_{session_id}/tb')
    best_model_path = Path(f'sessions/session_{session_id}/best_model')

    if cfg["save_rewards"]:
        cfg["save_path"] = f'sessions/session_{session_id}'

    # create environment
    env = SubprocVecEnv([make_env(i, cfg) for i in range(n_envs)])
    eval_env = vec_transpose.VecTransposeImage(env)

    # establish callbacks
    enable_callbacks = True
    callbacks = []

    if enable_callbacks:
        checkpoint_callback = CheckpointCallback(save_freq=n_steps, 
                                                 save_path=session_path, 
                                                 name_prefix='mai')
        
        evaluation_callback = EvalCallback(eval_env, 
                                           eval_freq=n_steps, 
                                           log_path=session_path, 
                                           best_model_save_path=best_model_path)

        callbacks.append(checkpoint_callback)
        callbacks.append(evaluation_callback)

    callbacks = CallbackList(callbacks)

    model = DQN('CnnPolicy', 
                env, 
                verbose=1, 
                buffer_size=10000,
                batch_size=256, 
                tensorboard_log=tb_path)
    
    # load pretrained model
    train_on_pretrained = False
    if train_on_pretrained:
        file_name = 'sessions/session_cd1f9/mai_51200_steps'
        model = DQN.load(file_name, env=env)
        model.verbose = 1
        model.batch_size = 256
        model.n_envs = n_envs
        model.tensorboard_log=tb_path


    model.learn(total_timesteps=n_steps*n_envs*1, callback=callbacks)

    # close environments
    env.close()
    eval_env.close()
