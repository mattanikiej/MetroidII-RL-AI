from random import randint

from metroid_env import MetroidGymEnv

from pyboy import PyBoy

from stable_baselines3 import PPO

import basic_config as bc
import memory_constants as c

import torch
from stable_baselines3.common.env_checker import check_env

pyboy = PyBoy('../ROMs/Metroid2.gb')
f = open("../states/post_start_screen.state", "rb")
pyboy.load_state(f)
f.close()

env = MetroidGymEnv(bc.config)
check_env(env)
model = PPO('CnnPolicy', env, verbose=1).learn(2)

# for i in range(10000):
#     pyboy.tick()