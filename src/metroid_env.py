from random import randint

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
            raise Exception("Config needs to be set for MetroidGymEnv. Check basic_config.py for structure")
        
        # load in config values
        self.action_frequency = config['action_frequency']
        self.states = config['states']
        self.rom_path = config['rom_path']
        self.seed = config['seed']
        self.max_steps = config['max_steps']
        self.window_type = config['window']

        # initial state is initialized in self.reset()
        self.initial_state = None

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
        self.pyboy = PyBoy(self.rom_path, window_type=self.window_type)

        self.screen = self.pyboy.botsupport_manager().screen()

        # set gym attributes
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.reward_range = (0, 15000)
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)

        # rewards are initialized during self.update_rewards() in self.reset()
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
            'armor_upgrade': 2,
            'beam_upgrade': 2,
            'metroids_remaining': 4,
            'deaths': 1
        }

        self.total_reward = 0

        self.previous_health = 0
        self.previous_missles = 0
        self.previous_armor_upgrade = 0
        self.previous_beam_upgrade = 0
        self.previous_metroids_remaining = 0
        self.deaths = 0

        self.steps_taken = 0
            
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

        reward_gain = self.update_rewards()

        self.steps_taken += 1

        terminated = self.check_if_done()

        return game_pixels, reward_gain, terminated, False, {}


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
        self.steps_taken = 0

        # choose random start state
        i = randint(0, len(self.states)-1)
        state = self.states[i]
        self.initial_state = state

        with open(state, "rb") as f:
            self.pyboy.load_state(f)

        # reset rewards
        self.previous_health = self.read_memory(mem.CURRENT_HP)
        self.previous_missles = self.read_memory(mem.CURRENT_MISSLES)
        self.previous_armor_upgrade = self.read_memory(mem.CURRENT_ARMOR_UPGRADE)
        self.previous_beam_upgrade = self.read_memory(mem.CURRENT_BEAM_UPGRADE)
        self.previous_metroids_remaining = self.read_memory(mem.GLOBAL_METROIDS_REMAINING)

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

            # check hp to see if game needs to be reset
            if self.samus_is_dead():
                self.deaths += 1
                with open(self.initial_state, "rb") as f:
                    self.pyboy.load_state(f)

                # reset rewards
                self.previous_health = self.read_memory(mem.CURRENT_HP)
                self.previous_missles = self.read_memory(mem.CURRENT_MISSLES)
                self.previous_armor_upgrade = self.read_memory(mem.CURRENT_ARMOR_UPGRADE)
                self.previous_beam_upgrade = self.read_memory(mem.CURRENT_BEAM_UPGRADE)
                self.previous_metroids_remaining = self.read_memory(mem.GLOBAL_METROIDS_REMAINING)

                self.update_rewards(reset=True)


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


    def update_rewards(self, reset=False):
        """
        Updates all of the rewards and returns the net reward gain

        :param reset (bool): If this is being called during reset, 
                             it takes the death into accoutn during the difference

        :return: (int)
        """
        self.rewards = {
            'health_pickup': self.get_health_pickup_reward(),
            'missle_pickup': self.get_missle_pickup_reward(),
            'armor_upgrade': self.get_armor_upgrade_reward(),
            'beam_upgrade': self.get_beam_upgrade_reward(),
            'metroids_remaining': self.get_metroids_remaining_reward(),
            'deaths': self.get_deaths_reward()
        }

        state_rewards = 0
        for reward in self.rewards:
            state_rewards += self.reward_weights[reward] * self.rewards[reward]

        reward_difference = state_rewards - self.total_reward
        self.total_reward = state_rewards

        # get total reward with one less death
        # punishes dying without resetting gain/ loss of other rewards
        if reset:
            self.total_reward -= self.rewards['deaths'] * self.reward_weights['deaths']
            self.total_reward += (self.rewards['deaths']+1) * self.reward_weights['deaths']

        return reward_difference


    def get_health_pickup_reward(self):
        """
        Checks memory and returns the current health of Samus

        :return: (int)
        """
        curr_health = self.read_memory(mem.CURRENT_HP)
        reward = curr_health - self.previous_health
        # don't overly punish getting hit
        if reward < 0:
            reward *= 0.1
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
        return -self.deaths


    def read_memory(self, address):
        """
        Gets the value at the given address and returns it

        :param address (hex): memory address to check

        :return: (int)
        """
        return self.pyboy.get_memory_value(address)


    def check_if_done(self):
        done = False
        if self.steps_taken >= self.max_steps:
            print(f"Total Rewards: {self.total_reward}")
            done = True
        return done
