import os
from rl_parsers.dpomdp import parse
import numpy as np
import gym
from gym.utils import seeding
from gym import spaces

class DPOMDP(gym.Env):
    """Environment specified by DPOMDP file"""
    def __init__(self, path, episodic=False, seed=None):
        debug=False
        #debug = True if 'skewed' in path else False
        self.episodic = episodic
        self.seed(seed)
        with open(path) as f:
            model = parse(f.read(), debug=debug)

        self.discount = model.discount
        self.agents = len(model.agents)
        self.state_space = spaces.Discrete(len(model.states))
        self.action_space = [spaces.Discrete(len(aAct)) for aAct in model.actions]  # Space for each agent
        self.observation_space = [spaces.Discrete(len(aObs)) for aObs in model.observations]  # Obs space for each agent
        self.reward_range = model.R.min(), model.R.max()
        if model.start is None:
            self.start = np.full(self.state_space.n, 1 / self.state_space.n)
        else:
            self.start = model.start

        # Start-state, agent actions, end state
        self.T = model.T.transpose(self.agents, *(np.arange(self.agents)), self.agents+1).copy()
        # Start, agent actions, end, agent observations
        self.O = np.stack([model.O] * self.state_space.n)
        # Start-state, agent actions, end state, agent observations
        self.R = model.R.transpose(self.agents, *(np.arange(self.agents)), self.agents+1, *(np.arange(self.agents)+self.agents+2)).copy()

        if episodic:
            self.D = model.reset.T.copy()  # only if episodic

        self.state = None

    def seed(self, seed):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self):
        self.state = self.np_random.multinomial(1, self.start).argmax().item()

    # Take an nparray or tuple of actions, not just 1
    def step(self, actions):
        assert self.state is not None, 'State has not been initialized'
        assert actions.shape[0] == self.agents, 'Must provide joint action'

        state1 = self.np_random.multinomial(
            1, self.T[self.state, (*actions)]).argmax().item()
        # Need to get the joint obs. The last indices are obs for agents 1, 2, etc
        # Flatten those indices (they sum to 1), draw a number, unflatten
        # So obs is a tuple of length (numAgents)
        obs = np.unravel_index(self.np_random.multinomial(1, self.O[self.state, (*actions), state1].flatten()).argmax(),
                               tuple([oSpace.n for oSpace in self.observation_space]))
        reward = self.R[self.state, (*actions), state1, (*obs)].item()

        if self.episodic:
            done = self.D[self.state, (*actions)]
        else:
            done = False

        if done:
            self.state = None
        else:
            self.state = state1

        return obs, reward, done, {}


if __name__=="__main__":
    pathToDPOMDPs = 'DPOMDPs'
    DPOMDPFileList = [os.path.join(pathToDPOMDPs, file) for file in os.listdir(pathToDPOMDPs)]
    for dpom in DPOMDPFileList:
        if 'tiger' in dpom or 'example' in dpom:
            continue
        testDPOMDP = DPOMDP(dpom, episodic=True)
        testDPOMDP.reset()
        testDPOMDP.step(np.arange(2))