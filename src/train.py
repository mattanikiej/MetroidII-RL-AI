from pathlib import Path
from uuid import uuid4

from stable_baselines3 import PPO
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
    # eval_env = vec_transpose.VecTransposeImage(env)

    # establish callbacks
    enable_callbacks = True
    callbacks = []

    if enable_callbacks:
        checkpoint_callback = CheckpointCallback(save_freq=n_steps, 
                                                 save_path=session_path, 
                                                 name_prefix='mai')
        
        # evaluation_callback = EvalCallback(eval_env, 
        #                                    eval_freq=n_steps, 
        #                                    log_path=session_path, 
        #                                    best_model_save_path=best_model_path)

        callbacks.append(checkpoint_callback)
        # callbacks.append(evaluation_callback)

    callbacks = CallbackList(callbacks)

    model = None
    n_epochs = 10
    train_on_pretrained = True
    if train_on_pretrained:
        file_name = 'sessions/session_158eb/mai_3276800_steps'
        model = PPO.load(file_name, env=env)
        model.verbose = 1
        model.n_epochs = n_epochs
        model.batch_size = 256
        model.n_steps = n_steps//8
        model.n_envs = n_envs
        model.rollout_buffer.reset()

    else:
        model = PPO('CnnPolicy', 
                    env, 
                    verbose=1, 
                    n_epochs=n_epochs, 
                    batch_size=256, 
                    n_steps=n_steps // 8, 
                    tensorboard_log=tb_path)

    learning_iters = 1
    for i in range(learning_iters):
        model.learn(total_timesteps=n_steps*n_envs*10, callback=callbacks)

    # close environments
    env.close()
    # eval_env.close()
