from random import randint

from metroid_env import MetroidGymEnv

from pyboy import PyBoy

import basic_config as bc

# pyboy = PyBoy('ROMs/Metroid2.gb')
# f = open("states/post_start_screen.state", "rb")
# pyboy.load_state(f)
# f.close()

env = MetroidGymEnv(bc.config)

for i in range(500):
    action = randint(0, 12)
    env.step(action)

env.close()
