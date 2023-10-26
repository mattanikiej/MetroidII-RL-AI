# std libs
import random

# data/ math libs
import numpy as np

# machine learning libs
from gymnasium import Env, spaces

# pyboy 
from pyboy import PyBoy
from pyboy.utils import WindowEvent

# src
import memory_constants as mem


class MetroidGymEnv(Env):
    """
    Gymasium environment to be used by the model
    """
    def __init__(self, config=None):
        """
        Constructor for MetroidGymEnv
        
        :param config (dict): configuration settings for the environment
        """
        # check a config was passed in
        if config is None:
            raise Exception("Config needs to be set for MetroidGymEnv")
        
        # load in config values
        self.action_frequency = config['action_frequency']
        self.initial_state = config['initial_state']
        self.rom_path = config['rom_path']
        self.seed = config['seed']

        # initialize movement
        self.valid_actions = [
            # move samus
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_ARROW_RIGHT,

            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_ARROW_RIGHT,

            # jump/ shoot
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,

            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,

            # toggle missles
            WindowEvent.PRESS_BUTTON_SELECT
        ]

        self.last_pressed = None

        # load in the emulator and game
        self.pyboy = PyBoy(self.rom_path)

        self.screen = self.pyboy.botsupport_manager().screen()

        # set gym attributes
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.reward_range = (0, 15000)
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)

        # set rewards
        self.rewards = {
            'health_pickup': 0,
            'missle_pickup': 0,
            'armor_upgrade': 0,
            'beam_upgrade': 0,
            'metroids_remaining': 0,
            'deaths': 0
        }

        self.reward_weights = {
            'health_pickup': 1,
            'missle_pickup': 1,
            'armor_upgrade': 1,
            'beam_upgrade': 1,
            'metroids_remaining': 1,
            'deaths': 1
        }

        self.total_reward = 0

        self.previous_health = 0
        self.previous_missles = 0
        self.previous_armor_upgrade = 0
        self.previous_beam_upgrade = 0
        self.previous_metroids_remaining = 0
        self.deaths = 0
            
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
        self.act(action)
        game_pixels = self.render()

        return game_pixels, 0, False, False, {}


    def reset(self, seed=None):
        """
        Resets the environment to an initial state, required before calling step. 
        Returns the first agent observation for an episode and information, 
        i.e. metrics, debug info.

        https://gymnasium.farama.org/api/env/

        :param seed (int): random state to use for rng

        :return: (ObsType), (dict)
        """
        self.seed = seed

        with open(self.initial_state, "rb") as f:
            self.pyboy.load_state(f)

        # reset rewards
        self.previous_health = self.read_memory(mem.CURRENT_HP)
        self.previous_missles = self.read_memory(mem.CURRENT_MISSLES)
        self.previous_armor_upgrade = self.read_memory(mem.CURRENT_ARMOR_UPGRADE)
        self.previous_beam_upgrade = self.read_memory(mem.CURRENT_BEAM_UPGRADE)
        self.previous_metroids_remaining = self.read_memory(mem.GLOBAL_METROIDS_REMAINING)
        self.deaths = 0

        self.update_rewards()

        return self.render(), {}


    def render(self):
        """
        Renders the environments to help visualise what the agent see, 
        examples modes are “human”, “rgb_array”, “ansi” for text.

        https://gymnasium.farama.org/api/env/

        :return: (list[int])
        """
        # get screen pixels values
        game_pixels = self.screen.screen_ndarray() # (144, 160, 3)
        return game_pixels


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
        # only WindowEvent.PRESS_BUTTON_SELECT should be immediately released
        select_pressed = False
        if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_SELECT:
            select_pressed = True

        # send action then tick self.action_frequency number of steps
        self.pyboy.send_input(self.valid_actions[action])

        for _ in range(self.action_frequency):
            # advance game 1 frame
            self.pyboy.tick()

            # release select if pressed
            if select_pressed:
                select_pressed = False
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_SELECT)


    def update_rewards(self):
        """
        Updates all of the rewards in the dictionary
        """
        self.rewards = {
            'health_pickup': self.get_health_pickup_reward(),
            'missle_pickup': self.get_missle_pickup_reward(),
            'armor_upgrade': self.get_armor_upgrade_reward(),
            'beam_upgrade': self.get_beam_upgrade_reward(),
            'metroids_remaining': self.get_metroids_remaining_reward(),
            'deaths': self.get_deaths_reward()
        }

        for reward in self.rewards:
            self.total_reward += self.reward_weights[reward] * self.rewards[reward]


    def get_health_pickup_reward(self):
        """
        Checks memory and returns the current health of Samus

        :return: (int)
        """
        curr_health = self.read_memory(mem.CURRENT_HP)
        reward = curr_health - self.previous_health
        return reward


    def get_missle_pickup_reward(self):
        """
        Checks memory and returns the difference of missles

        :return: (int)
        """
        curr_missles = self.read_memory(mem.CURRENT_MISSLES)
        reward = curr_missles - self.previous_missles

        # don't punish ai for using missles
        if reward < 0:
            reward = 0

        return reward


    def get_armor_upgrade_reward(self):
        """
        Checks memory and returns the armor reward

        :return: (int)
        """
        curr_armor = self.read_memory(mem.CURRENT_ARMOR_UPGRADE)
        reward = curr_armor - self.previous_armor_upgrade
        return reward


    def get_beam_upgrade_reward(self):
        """
        Checks memory and returns the beam reward

        :return: (int)
        """
        curr_beam = self.read_memory(mem.CURRENT_BEAM_UPGRADE)
        reward = curr_beam - self.previous_beam_upgrade
        return reward


    def get_metroids_remaining_reward(self):
        """
        Check memory and return the metroids reward

        :return: (int)
        """
        curr_metroids = self.read_memory(mem.GLOBAL_METROIDS_REMAINING)
        reward = self.previous_metroids_remaining - curr_metroids
        return reward


    def get_deaths_reward(self):
        """
        Gets the amount of times the ai has dies. Wrapper for self.deaths

        :return: (int)
        """
        return self.deaths


    def read_memory(self, address):
        """
        Gets the value at the given address and returns it

        :param address (hex): memory address to check

        :return: (int)
        """
        return self.pyboy.get_memory_value(address)

        
