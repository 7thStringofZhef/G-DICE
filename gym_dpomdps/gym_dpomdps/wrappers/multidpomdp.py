import gym
import numpy as np
from ..envs import DPOMDP

# For dpomdps
#   Only difference is that actions and observations now have an additional dimension for each agent at the end
class MultiDPOMDP(gym.Wrapper):
    def __init__(self, env, nTrajectories):
        assert isinstance(env, DPOMDP)
        super(MultiDPOMDP, self).__init__(env)
        self.nAgents = env.agents
        self.nTrajectories = nTrajectories
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
    #   actions: (nTrajectories, nAgents) nparray of action indices. Each row represents a joint action
    # Output:
    #   obs: (nTrajectories, nAgents) nparray of observation indices. -1 for completed trajectories
    #   rewards: (nTrajectories) nparray of rewards
    #   done: (nTrajectories) nparray of whether a particular trajectory is done
    def step(self, actions):
        # Make sure numpy array is appropriate size (nTrajectories, nAgents)
        assert actions.shape[0] == self.nTrajectories
        assert actions.shape[1] == self.nAgents

        # For each trajectory that is done, return nothing
        doneIndices = np.nonzero(self.state == -1)[0]
        notDoneIndices = np.nonzero(self.state != -1)[0]

        # Blank init
        newStates = np.zeros(self.nTrajectories, dtype=np.int32)
        obs = np.zeros((self.nTrajectories, self.nAgents), dtype=np.int32)
        rewards = np.zeros(self.nTrajectories, dtype=np.float64)
        done = np.ones(self.nTrajectories, dtype=bool)

        # Reduced list based on which workers are done. If env is not episodic, this will still work
        validStates = self.state[notDoneIndices]
        validActions = actions[notDoneIndices, :]
        validActionIndices = tuple(aAct.flatten() for aAct in np.split(validActions, self.nAgents, axis=1))
        validNewStates = np.array([self.np_random.multinomial(1, p).argmax()
                                   for p in self.env.T[validStates, (*validActionIndices)]])
        validObs = np.array([self.np_random.multinomial(1, p.flatten()).argmax()
                             for p in self.env.O[validStates, (*validActionIndices), validNewStates]])
        validObs = np.unravel_index(validObs, self.env.O.shape[-self.nAgents:])  # Convert back to indices for each observation
        validRewards = np.array(self.env.R[validStates, (*validActionIndices), validNewStates, (*validObs)])
        if self.env.episodic:
            done[notDoneIndices] = self.env.D[self.state, (*validActionIndices)]
        else:
            done *= False

        newStates[notDoneIndices], newStates[doneIndices] = validNewStates, -1
        obs[notDoneIndices, :], obs[doneIndices, :] = np.array(validObs).T, -1
        rewards[notDoneIndices], rewards[doneIndices] = validRewards, 0.0
        self.state = newStates

        return obs, rewards, done, {}
