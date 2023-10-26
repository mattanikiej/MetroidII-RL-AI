import faulthandler

from metroid_env import MetroidGymEnv

from stable_baselines3 import PPO

import basic_config as bc
import memory_constants as c

from stable_baselines3.common.env_checker import check_env

# pyboy = PyBoy('../ROMs/Metroid2.gb')
# f = open("../states/post_start_screen.state", "rb")
# pyboy.load_state(f)
# f.close()

faulthandler.enable()

ep_length = 100
bc.config["max_steps"] = ep_length
env = MetroidGymEnv(bc.config)
check_env(env)
model = PPO('CnnPolicy', env, verbose=1, batch_size=ep_length, n_steps=ep_length)

for i in range(2):
    if i == 1:
        print("2nd ITERATION LEARNING")
    model.learn(total_timesteps=ep_length)

print("Done Learning")
env.close()



# for i in range(10000):
#     pyboy.tick()
#     print(pyboy.get_memory_value(c.CURRENT_HP))