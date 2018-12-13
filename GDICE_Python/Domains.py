from gym_pomdps import list_pomdps, POMDP
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

# States, observations, rewards, actions, dones are now lists or np arrays
class MultiActionPOMDP(gym.Wrapper):
    def __init__(self, env, numTrajectories):
        assert isinstance(env, POMDP)
        super().__init__(env)
        self.numTrajectories = numTrajectories
        self.reset()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self):
        if self.env.start is None:
            self.state = self.np_random.randint(
                self.state_space.n, size=self.numTrajectories)
        else:
            self.state = self.np_random.multinomial(
                1, self.env.start/np.sum(self.env.start), size=self.numTrajectories).argmax(1)

    # Step given an nparray or list of actions
    # If actions is a scalar, applies to all
    def step(self, actions):
        # Scalar action given, apply to all
        if np.isscalar(actions):
            actions = np.full(self.numTrajectories, actions, dtype=np.int32)

        # Tuple of (action, index) given, step for one worker only
        if isinstance(actions, tuple) and len(actions) == 2:
            return self._stepForSingleWorker(int(actions[0]), int(actions[1]))

        # For each agent that is done, return nothing
        doneIndices = np.nonzero(self.state == -1)[0]
        notDoneIndices = np.nonzero(self.state != -1)[0]

        # Blank init
        newStates = np.zeros(self.numTrajectories, dtype=np.int32)
        obs = np.zeros(self.numTrajectories, dtype=np.int32)
        rewards = np.zeros(self.numTrajectories, dtype=np.float64)
        done = np.ones(self.numTrajectories, dtype=bool)

        # Reduced list based on which workers are done. If env is not episodic, this will still work
        validStates = self.state[notDoneIndices]
        validActions = actions[notDoneIndices]
        validNewStates = np.array([self.np_random.multinomial(1, p).argmax() for p in self.env.T[validStates, validActions]])
        validObs = np.array([self.np_random.multinomial(1, p).argmax() for p in self.env.O[validStates, validActions, validNewStates]])
        validRewards = np.array(self.env.R[validStates, validActions, validNewStates, validObs])
        if self.env.episodic:
            done[notDoneIndices] = self.env.D[self.state, actions]
        else:
            done *= False

        newStates[notDoneIndices], newStates[doneIndices] = validNewStates, -1
        obs[notDoneIndices], obs[doneIndices] = validObs, -1
        rewards[notDoneIndices], rewards[doneIndices] = validRewards, 0.0
        self.state = newStates

        return obs, rewards, done, {}

    # If multiprocessing, each worker will provide its trajectory index and desired action
    def _stepForSingleWorker(self, action, index):
        currState = self.state[index]

        # If this worker's episode is finished, return nothing
        if currState is None:
            return -1, 0.0, True, {}

        newState = self.np_random.multinomial(1, self.env.T[currState, action]).argmax()
        obs = self.np_random.multinomial(1, self.env.O[currState, action, newState]).argmax()
        reward = self.env.R[currState, action, newState, obs]
        if self.env.episodic:
            done = self.env.D[currState, action]
        else:
            done = False

        if done:
            self.state[index] = -1
        else:
            self.state[index] = newState

        return obs, reward, done, {}

# For DPOMDPs
#   Only difference is that everything now has an additional dimension for the number of agents (last dimension?)
class MultiActionDPOMDP(MultiActionPOMDP):
    def __init__(self, env, numTrajectories):
        assert isinstance(env, POMDP)
        super(MultiActionDPOMDP, self).__init__(env, numTrajectories)
        self.numAgents = env.numAgents  # ***Need to see what this will be
        self.reset()

    # Step given an nparray of actions (numAgents, numTrajectories, numActions)
    def step(self, actions):
        # Scalar action given, apply to all
        if np.isscalar(actions):
            actions = np.full(self.numTrajectories, actions, dtype=np.int32)

        # Tuple of (actions, trajectoryIndex) given, step for one worker only
        if isinstance(actions, tuple) and len(actions) == 2:
            return self._stepForSingleWorker(actions[0], int(actions[1]))

        # For each agent that is done, return nothing
        doneIndices = np.nonzero(self.state == -1)[0]
        notDoneIndices = np.nonzero(self.state != -1)[0]

        # Blank init
        newStates = np.zeros(self.numTrajectories, dtype=np.int32)
        obs = np.zeros(self.numTrajectories, dtype=np.int32)
        rewards = np.zeros(self.numTrajectories, dtype=np.float64)
        done = np.ones(self.numTrajectories, dtype=bool)

        # Reduced list based on which workers are done. If env is not episodic, this will still work
        validStates = self.state[notDoneIndices]
        validActions = actions[notDoneIndices]
        validNewStates = np.array([self.np_random.multinomial(1, p).argmax() for p in self.env.T[validStates, validActions]])
        validObs = np.array([self.np_random.multinomial(1, p).argmax() for p in self.env.O[validStates, validActions, validNewStates]])
        validRewards = np.array(self.env.R[validStates, validActions, validNewStates, validObs])
        if self.env.episodic:
            done[notDoneIndices] = self.env.D[self.state, actions]
        else:
            done *= False

        newStates[notDoneIndices], newStates[doneIndices] = validNewStates, -1
        obs[notDoneIndices], obs[doneIndices] = validObs, -1
        rewards[notDoneIndices], rewards[doneIndices] = validRewards, 0.0
        self.state = newStates

        return obs, rewards, done, {}

    # If multiprocessing, each worker will provide its trajectory index and desired action
    def _stepForSingleWorker(self, action, index):
        currState = self.state[index]

        # If this worker's episode is finished, return nothing
        if currState is None:
            return np.full(self.numAgents, -1, dtype=np.int32), \
                   np.zeros(self.numAgents, dtype=np.float64),\
                   np.ones(self.numAgents, dtype=bool), {}

        newState = self.np_random.multinomial(1, self.env.T[currState, tuple(np.split(action, action.shape[0]))]).argmax()
        obs = self.np_random.multinomial(1, self.env.O[currState, tuple(np.split(action, action.shape[0])), newState]).argmax()
        reward = self.env.R[currState, tuple(np.split(action, action.shape[0])), newState, obs]
        if self.env.episodic:
            done = self.env.D[currState, tuple(np.split(action, action.shape[0]))]
        else:
            done = False

        if done:
            self.state[index] = -1
        else:
            self.state[index] = newState

        return obs, reward, done, {}


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