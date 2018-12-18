import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import IntEnum

#   ######
#   #A   #
#   # #  #
#   #   T#
#   ######

# A - the agent (UAV)
# T - the target to be detected

# actions: n s e w
# observations: wall_left wall_right target_left target_right neither

# The default reward/penalty is -0.1
# If the target detected then the reward is 1

# Starting state: the UAV is located in the upper-left corner
# Episode termination:
#    The UAV detected the target
#    Episode length is greater than 50

class Action(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

class Observation(IntEnum):
    WALL_LEFT = 0
    WALL_RIGHT = 1
    TARGET_LEFT = 2
    TARGET_RIGHT = 3
    NEITHER = 4

class UAVSimpleDomain(gym.Env):
    def __init__(self, seed=None):
        self.episodic = True
        self.seed(seed)

        self.agents = 1
        self.discount = 0.99
        self.grid = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        self.n_rows = self.grid.shape[0]
        self.n_columns = self.grid.shape[1]

        self.action_space = spaces.Discrete(4)  # 4 actions
        self.observation_space = spaces.Discrete(5)  # 5 observations
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self):
        uav_location = (0, 0)
        target_location = (2, 3)
        self.state = (uav_location, target_location)

    def step(self, action, state=None):
        # The state argument allows us to simulate many states in parallel
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # In GDICE we simulate many states in parallel
        if state is None:
            state = self.state

        # Extract the state components
        uav_location, target_location = state

        # Update the UAV location
        if action == Action.NORTH:
            new_location = (uav_location[0] - 1, uav_location[1])
        elif action == Action.SOUTH:
            new_location = (uav_location[0] + 1, uav_location[1])
        elif action == Action.EAST:
            new_location = (uav_location[0], uav_location[1] + 1)
        elif action == Action.WEST:
            new_location = (uav_location[0], uav_location[1] - 1)

        # If the new location is out of boundary, then stay in place
        if new_location[0] >= 0 and new_location[0] < self.n_rows and \
            new_location[1] >= 0 and new_location[1] < self.n_columns:
            uav_location = new_location

        # Update the state
        new_state = (uav_location, target_location)
        if state is None:
            self.state = new_state

        # Check if the UAV reached the target
        if uav_location == target_location:
            reward = 1.0
            done = True
        else:
            reward = -0.1
            done = False

        # Generate a new observation according to the UAV's location
        if uav_location[1] == 0 or (uav_location[1] > 0 and
            self.grid[uav_location[0], uav_location[1] - 1] == 1):
            obs = Observation.WALL_LEFT
        elif uav_location[1] == self.n_columns - 1 or (uav_location[1] < self.n_columns - 1 and
            self.grid[uav_location[0], uav_location[1] + 1] == 1):
            obs = Observation.WALL_RIGHT
        elif uav_location[0] == target_location[0] and \
            uav_location[1] == target_location[1] + 1:
            obs = Observation.TARGET_LEFT
        elif uav_location[0] == target_location[0] and \
            uav_location[1] == target_location[1] - 1:
            obs = Observation.TARGET_RIGHT
        else:
            obs = Observation.NEITHER

        return obs, reward, done, { 'new_state': new_state }
