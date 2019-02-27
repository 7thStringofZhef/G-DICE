from gym_pomdps import POMDP
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

# States, observations, rewards, actions, dones are now lists or np arrays
class MultiPOMDP(gym.Wrapper):
    def __init__(self, env, numTrajectories):
        assert isinstance(env, POMDP)
        super().__init__(env)
        self.nTrajectories = numTrajectories
        self.reset()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self):
        if self.env.start is None:
            self.state = self.np_random.randint(
                self.state_space.n, size=self.nTrajectories)
        else:
            self.state = self.np_random.multinomial(
                1, self.env.start/np.sum(self.env.start), size=self.nTrajectories).argmax(1)

    # Step given an nparray of actions
    # Input:
    #   actions: nparray, apply to all states
    # Output:
    #   obs: nparray of observation indices for each trajectory. -1 for completed trajectories
    #   rewards: nparray of rewards for each trajectory
    #   done: nparray of whether a particular trajectory is done
    def step(self, actions):
        # Make sure numpy array is appropriate size
        assert actions.shape[0] == self.nTrajectories

        # For each agent that is done, return nothing
        doneIndices = np.nonzero(self.state == -1)[0]
        notDoneIndices = np.nonzero(self.state != -1)[0]

        # Blank init. Invalid states/obs, 0 reward, all done
        newStates = np.zeros(self.nTrajectories, dtype=np.int32)
        obs = np.zeros(self.nTrajectories, dtype=np.int32)
        rewards = np.zeros(self.nTrajectories, dtype=np.float64)
        done = np.ones(self.nTrajectories, dtype=bool)

        # Reduced list based on which workers are done. If env is not episodic, this will still work
        validStates = self.state[notDoneIndices]
        validActions = actions[notDoneIndices]
        validNewStates = np.array([self.np_random.multinomial(1, p).argmax() for p in self.env.T[validStates, validActions]])
        validObs = np.array([self.np_random.multinomial(1, p).argmax() for p in self.env.O[validStates, validActions, validNewStates]])
        validRewards = np.array(self.env.R[validStates, validActions, validNewStates, validObs])
        if self.env.episodic:
            done[notDoneIndices] = self.env.D[validStates, actions]
        else:
            done *= False

        newStates[notDoneIndices], newStates[doneIndices] = validNewStates, -1
        obs[notDoneIndices], obs[doneIndices] = validObs, -1
        rewards[notDoneIndices], rewards[doneIndices] = validRewards, 0.0
        self.state = newStates

        return obs, rewards, done, {}


class PackageDeliveryEnvironment(gym.Env):
    # Map: Base1 (B1), Base2 (B2), Rendezvous (R), Delivery1 (D1), Delivery2 (D2)
    # 0  0 D2 0
    # B1 0 0  R
    # 0  0 0  0
    # B2 0 D1 0
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
        return [seed_]

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

"""
In the battleship POMDP, 5 ships are placed at random into a 10 × 10 grid, subject to the
constraint that no ship may be placed adjacent or diagonally adjacent to another ship. Each
ship has a different size of 5 × 1, 4 × 1, 3 × 1 and 2 × 1 respectively. The goal is to find
and sink all ships. However, the agent cannot observe the location of the ships. Each step,
the agent can fire upon one cell of the grid, and receives an observation of 1 if a ship was
hit, and 0 otherwise. There is a -1 reward per time-step, a terminal reward of +100 for
hitting every cell of every ship, and there is no discounting (γ = 1). It is illegal to fire twice
on the same cell. If it was necessary to fire on all cells of the grid, the total reward is 0;
otherwise the total reward indicates the number of steps better than the worst case. There
are 100 actions, 2 observations, and approximately 1018 states in this challenging POMDP.
Particle invigoration is particularly important in this problem. Each local transformation
applied one of three transformations: 2 ships of different sizes swapped location; 2 smaller
ships were swapped into the location of 1 larger ship; or 1 to 4 ships were moved to a new
location, selected uniformly at random, and accepted if the new configuration was legal.
Without preferred actions, all legal actions were considered. When preferred actions were
used, impossible cells for ships were deduced automatically, by marking off the diagonally
adjacent cells to each hit. These cells were never selected in the tree or during rollouts. The
performance of POMCP, with and without preferred actions, is shown in Figure 2
"""

# 4 ships (5x1, 4x1, 3x1, 2x1)
# 10*10 grid, no adjacent ships
class BattleshipEnvironment(gym.env):
    pass