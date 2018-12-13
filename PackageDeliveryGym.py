from gym import spaces, Env
from gym.utils import seeding
import numpy as np

class PackageDeliveryEnvironment(gym.Env):
        #Map: Base1 (B1), Base2 (B2), Rendezvous (R), Delivery1 (D1), Delivery2 (D2)
        #0  0 D2 0
        #B1 0 0  R
        #0  0 0  0
        #B2 0 D1 0
    def __init__(self, seed=None):
        self.episodic = False
        self.seed(seed)

        self.agents = 2
        self.discount = 0.99
        self.state_space = spaces.Discrete(6)  # ***What should this be?
        self.action_space = spaces.Discrete(13)  # 13 TMAs
        self.observation_space = spaces.Discrete(13)  # 13 observations
        self.reward_range = 0.0, 3.0
        self.start = None  # ***Depends on state
        self.T = None  # ***Also depends on state
        self.O = None  # ***Also depends on state
        self.R = None  # ***Also depends on state

        self.D = False  # Not episodic

        self.state = None

    def seed(self, seed):
        self.np_random, seed_ = seeding.np_random(seed)
        return[seed_]

    def reset(self):
        self.state = self.np_random.multinomial(1, self.start).argmax().item()

    def step(self, action):
        assert self.state is not None
        state1 = self.np_random.multinomial(
            1, self.T[self.state, action]).argmax().item()
        obs = self.np_random.multinomial(
            1, self.O[self.state, action, state1]).argmax().item()
        reward = self.R[self.state, action, state1, obs].item()

        done = False
        self.state = state1

        return obs, reward, done, {}



