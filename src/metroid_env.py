from random import randint
from uuid import uuid4
from pathlib import Path
import math

import numpy as np
import pandas as pd

from gymnasium import Env, spaces

from pyboy import PyBoy
from pyboy.utils import WindowEvent

import memory_constants as mem
import checkpoint_path as chk


class MetroidGymEnv(Env):
    """
    Gymnasium environment to be used by the model
    """
    def __init__(self, config=None):
        """
        Constructor for MetroidGymEnv
        
        :param config (dict): configuration settings for the environment
        """
        # check a config was passed in
        if config is None:
            raise Exception("Config needs to be set for MetroidGymEnv. Check basic_config.py for structure")
        
        # load in config values
        self.action_frequency = config['action_frequency']
        self.states = config['states']
        self.rom_path = config['rom_path']
        self.seed = config['seed']
        self.max_steps = config['max_steps']
        self.window_type = config['window']
        self.save_rewards = config['save_rewards']
        self.save_path = None if not self.save_rewards else config['save_path']

        self.id = str(uuid4())[:5]

        # initial state is initialized in self.reset()
        self.initial_state = None

        # initialize movement
        self.valid_actions = [
            # move samus
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_ARROW_RIGHT,
            
            # jump/ shoot
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,

            # toggle missiles
            WindowEvent.PRESS_BUTTON_SELECT
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_SELECT
        ]

        self.last_pressed = None

        # load in the emulator and game
        self.pyboy = PyBoy(self.rom_path, window_type=self.window_type)


        # set gym attributes
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.reward_range = (-math.inf, math.inf)
        # observation is current frame and previous frame to give cnn sense of movement
        self.obs_shape = (144, 160, 3)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)

        # initialized in self.reset()
        self.previous_frame = None # (144, 160)

        # info for target distance reward
        self.reached_target = False
        self.target_screen_coord = (1,1) 
        self.max_dist = 0

        # rewards are initialized during self.update_rewards() in self.reset()
        self.rewards = {}

        # weights are all > 0
        self.reward_weights = {
            'health_pickup': 10,
            'missile_pickup': 10,
            'armor_upgrade': 50,
            'beam_upgrade': 50,
            'metroids_remaining': 200,
            'enemies_killed': 10,
            'exploration': 1,
            'target_distance': 2,
            'target_reached': 10,
            'checkpoint_passed': 10,

            'deaths': 1,
            'damage_taken': 1
        }
        
        self.rewards_df = None if not self.save_rewards else pd.DataFrame(self.rewards, index=[0])
        self.rewardw_df = None if not self.save_rewards else pd.DataFrame(self.reward_weights, index=[0])

        self.total_reward = 0

        self.previous_health = 0
        self.previous_missiles = 0
        self.previous_armor_upgrade = 0
        self.previous_beam_upgrade = 0
        self.previous_metroids_remaining = 0
        self.previous_sfx = 0
        self.previous_checkpoint = (0,0)

        self.enemies_killed = 0

        self.explored_coordinates = {}

        self.deaths = 0

        self.steps_taken = 0

        self.resets = -1

        if self.save_rewards:
            self.init_save_file()
            
        # start the game from initial state
        self.reset()


    def step(self, action):
        """
        Updates an environment with actions returning the next agent observation, 
        the reward for taking that actions, if the environment has terminated or 
        truncated due to the latest action and information from the environment 
        about the step, i.e. metrics, debug info.

        https://gymnasium.farama.org/api/env/

        :param action (actType): action the model selected to run

        :return: (ObsType), (SupportsFloat), (bool), (bool), (dict)
        """
        self.steps_taken += 1
        self.act(action)

        obs = self.render()
        reward_gain = self.update_rewards()
        terminated = self.check_if_done()

        return obs, reward_gain, terminated, False, {}


    def reset(self, seed=None):
        """
        Resets the environment to an initial state, required before calling step. 
        Returns the first agent observation for an episode and information, 
        i.e. metrics, debug info.

        https://gymnasium.farama.org/api/env/

        :param seed (int): random state to use for rng

        :return: (ObsType), (dict)
        """
        self.resets += 1
        self.seed = seed
        self.steps_taken = 0

        # choose random start state only when env is initialized
        if self.resets == 0:
            i = randint(0, len(self.states)-1)
            state = self.states[i]
            self.initial_state = state

        with open(self.initial_state, "rb") as f:
            self.pyboy.load_state(f)

        # reset rewards
        self.previous_health = self.read_memory(mem.CURRENT_HP)
        self.previous_missiles = self.read_memory(mem.CURRENT_MISSILES)
        self.previous_armor_upgrade = self.read_memory(mem.CURRENT_ARMOR_UPGRADE)
        self.previous_beam_upgrade = self.read_memory(mem.CURRENT_BEAM_UPGRADE)
        self.previous_metroids_remaining = self.read_memory(mem.GLOBAL_METROIDS_REMAINING)
        self.previous_sfx = self.read_memory(mem.SFX_PLAYING)
        self.previous_checkpoint = (self.read_memory(mem.PREV_SAMUS_X_SCREEN), 
                                    self.read_memory(mem.PREV_SAMUS_Y_SCREEN))

        self.enemies_killed = 0

        self.explored_coordinates = {}

        self.reached_target = False
        x = self.read_memory(mem.PREV_SAMUS_X_SCREEN)
        y = self.read_memory(mem.PREV_SAMUS_Y_SCREEN)
        self.max_dist = math.dist((x, y), self.target_screen_coord)

        self.update_rewards()

        screen = self.pyboy.botsupport_manager().screen()
        self.previous_frame = screen.screen_ndarray()[:, :, 0]

        return self.render(), {}


    def render(self):
        """
        Renders the environments to help visualise what the agent see, 
        examples modes are “human”, “rgb_array”, “ansi” for text.

        https://gymnasium.farama.org/api/env/

        :return: (list[int])
        """
        # get screen pixels values
        screen = self.pyboy.botsupport_manager().screen()
        # game is grayscale so only the top dimension from rgb array is needed
        frame_pixels = screen.screen_ndarray() # (144, 160, 3)

        # obs = np.array([frame_pixels, self.previous_frame])
        # obs = np.reshape(obs, self.obs_shape)

        return frame_pixels


    def close(self):
        """
        Closes the environment, important when external software is used, 
        i.e. pygame for rendering, databases

        https://gymnasium.farama.org/api/env/
        """
        self.pyboy.stop()


    def act(self, action):
        """
        Sends the given action to the emulator

        :param action (actType): action to send to pyboy
        """

        # send action then tick self.action_frequency number of steps
        self.pyboy.send_input(self.valid_actions[action])

        for i in range(self.action_frequency):
            # advance game 1 frame
            self.pyboy.tick()

            # check if enemy has died
            if self.has_enemy_died():
                self.enemies_killed += 1

            # check hp to see if game needs to be reset
            if self.samus_is_dead():
                self.deaths += 1
                self.reset()

            # get previous frame before next step
            if i == 2 - self.action_frequency:
                screen = self.pyboy.botsupport_manager().screen()
                self.previous_frame = screen.screen_ndarray()[:, :, 0]

        # release button
        self.pyboy.send_input(self.release_actions[action])


    def has_enemy_died(self):
        """
        Checks the memory to see if the enemy dying sfx started this frame.
        Returns True if it has, False otherwise.
        
        :return: (bool)
        """
        enemy_died = False
        sfx = self.read_memory(mem.SFX_PLAYING)
        if sfx == mem.ENEMY_KILLED_SFX and sfx != self.previous_sfx:
            enemy_died = True

        # update previous frame sfx
        self.previous_sfx = sfx

        return enemy_died


    def samus_is_dead(self):
        """
        Checks if Samus' hp is 0 and the game has reset

        :return: (bool)
        """
        dead = False
        health = self.pyboy.get_memory_value(mem.CURRENT_HP)
        if health <= 0:
            dead = True
        return dead


    def check_if_done(self):
        """
        Check if the max number of steps has been reached
        """
        done = False
        if self.steps_taken >= self.max_steps:
            print(f"Total Rewards: {self.total_reward}")
            
            if self.save_rewards:

                self.rewards_df = pd.DataFrame(self.rewards, index=[0])
                
                self.save_rewards_csv()

            done = True
        return done


    def init_save_file(self):
        """
        Initialize csv file paths for the instance
        """
        # create the path
        p = Path(self.save_path + f'/{self.id}')
        p.mkdir(parents=True, exist_ok=True)

        r = self.save_path + f'/{self.id}/rewards.csv.gz'
        rw = self.save_path + f'/{self.id}/reward_weights.csv.gz'
        self.save_path = (Path(r), Path(rw))

        self.rewards_df.to_csv(self.save_path[0], compression='gzip', mode='a', index=False)
        self.rewardw_df.to_csv(self.save_path[1], compression='gzip', mode='a', index=False)


    def save_rewards_csv(self):
        """
        Saves reward info to csv
        """
        self.rewards_df.to_csv(self.save_path[0], 
                               compression='gzip', 
                               mode='a', 
                               header=False, 
                               index=False)
        

    def update_rewards(self, reset=False):
        """
        Updates all of the rewards and returns the net reward gain

        :return: (int)
        """
        # all rewards are > 0 and all punishments are < 0
        self.rewards = {
            'health_pickup': self.get_health_pickup_reward(),
            'missile_pickup': self.get_missile_pickup_reward(),
            'armor_upgrade': self.get_armor_upgrade_reward(),
            'beam_upgrade': self.get_beam_upgrade_reward(),
            'metroids_remaining': self.get_metroids_remaining_reward(),
            'enemies_killed': self.get_enemies_killed_reward(),
            # 'exploration': self.get_exploration_reward(),
            # 'target_distance': self.get_target_distance_reward(),
            # 'target_reached': self.get_target_reached_reward(),
            'checkpoint_passed': self.get_checkpoint_passed_reward(),

            # 'deaths': self.get_deaths_punishment(),
            # 'damage_taken': self.get_damage_taken_punishment()
        }

        state_reward = 0
        for rwname in self.rewards:
            reward = self.reward_weights[rwname] * self.rewards[rwname]
            state_reward += reward

        reward_difference = state_reward - self.total_reward
        self.total_reward = state_reward

        return reward_difference


    def get_health_pickup_reward(self):
        """
        Checks memory and returns 1 if there are more missiles
        :return: (int)
        """
        curr_health = self.read_memory(mem.CURRENT_HP)

        reward = 0
        if curr_health > self.previous_health:
            reward = 1

        self.previous_health = curr_health

        return reward


    def get_missile_pickup_reward(self):
        """
        Checks memory and returns 1 if there are more missiles
        :return: (int)
        """
        curr_missiles = self.read_memory(mem.CURRENT_MISSILES)

        reward = 0
        if curr_missiles > self.previous_missiles:
            reward = 1

        self.previous_missiles = curr_missiles

        return reward


    def get_armor_upgrade_reward(self):
        """
        Checks memory and returns the armor reward if new armor was obtained

        :return: (int)
        """
        curr_armor = self.read_memory(mem.CURRENT_ARMOR_UPGRADE)

        reward = 0
        if curr_armor != self.previous_armor_upgrade:
            reward = 1

        self.previous_armor_upgrade = curr_armor

        return reward


    def get_beam_upgrade_reward(self):
        """
        Checks memory and returns the beam reward if new beam was obtained

        :return: (int)
        """
        curr_beam = self.read_memory(mem.CURRENT_BEAM_UPGRADE)
        
        reward = 0
        # check if beam is different, and not just switched to/from missiles
        if curr_beam != self.previous_beam_upgrade:
            reward = 1

        self.previous_beam_upgrade = curr_beam

        return reward


    def get_metroids_remaining_reward(self):
        """
        Check memory and return the metroids reward

        :return: (int)
        """
        curr_metroids = self.read_memory(mem.GLOBAL_METROIDS_REMAINING)
        reward = self.previous_metroids_remaining - curr_metroids
        return reward
    
    
    def get_enemies_killed_reward(self):
        """
        Gets the amount of enemies killed by the ai. Wrapper for self.enemies_killed

        :return: (int)
        """
        return self.enemies_killed


    def get_exploration_reward(self):
        """
        Checks to see if coordinates have been explored.
        Returns the number of unique coordinates explored

        :return: (int)
        """
        reward = 0

        # get screen x and y coordinates
        x = self.read_memory(mem.PREV_SAMUS_X_SCREEN)
        y = self.read_memory(mem.PREV_SAMUS_Y_SCREEN)

        # check if this pixel has been explored
        if x in self.explored_coordinates:
            self.explored_coordinates[x].add(y)        
        else:
            self.explored_coordinates[x] = set([y])

        # add all unique coordinates
        for xc in self.explored_coordinates:
            reward += len(self.explored_coordinates[xc])

        return reward


    def get_target_distance_reward(self):
        """
        Gets the distance from Samus to the target and returns the reward

        :return: (int)
        """
        x = self.read_memory(mem.PREV_SAMUS_X_SCREEN)
        y = self.read_memory(mem.PREV_SAMUS_Y_SCREEN)

        dist = math.dist((x, y), self.target_screen_coord)
        
        # reward increases as you get closer
        reward = self.max_dist - dist
        return reward


    def get_target_reached_reward(self):
        """
        Returns a reward if the target was reached, and only once

        :return: (int)
        """
        reward = 0

        x = self.read_memory(mem.PREV_SAMUS_X_SCREEN)
        y = self.read_memory(mem.PREV_SAMUS_Y_SCREEN)

        dist = math.dist((x, y), self.target_screen_coord)

        if not self.reached_target and dist == 0:
            self.reached_target = True
            reward = 1

        return reward  


    def get_checkpoint_passed_reward(self):
        """
        Returns a reward if passed the next checkpoint

        :return: (int)
        """
        reward = 0
        next_checkpoint = chk.checkpoints[self.previous_checkpoint]

        x = self.read_memory(mem.PREV_SAMUS_X_SCREEN)
        y = self.read_memory(mem.PREV_SAMUS_Y_SCREEN)

        curr = (x,y)
        if curr[0] == next_checkpoint[0] and curr[1] == next_checkpoint[1]:
            self.previous_checkpoint = curr
            reward = 1

        return reward


    def get_deaths_punishment(self):
        """
        Gets the amount of times the ai has dies. Wrapper for self.deaths

        :return: (int)
        """
        return -self.deaths


    def get_damage_taken_punishment(self):
        """
        Checks memory and returns the negative health difference of Samus

        :return: (int)
        """
        curr_health = self.read_memory(mem.CURRENT_HP)

        reward = 0
        if curr_health < self.previous_health:
            reward = 1

        self.previous_health = curr_health

        return reward


    def read_memory(self, address):
        """
        Gets the value at the given address and returns it

        :param address (hex): memory address to check

        :return: (int)
        """
        return self.pyboy.get_memory_value(address)
