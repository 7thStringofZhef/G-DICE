import gym
import numpy as np

# A class that wraps a gym environment in which states, observations, rewards, actions and dones
# are lists or np arrays, so multiple trajectories can be simulated in parallel

from copy import deepcopy

class GDICEEnvWrapper(gym.Wrapper):
    def __init__(self, env, numTrajectories):
        super().__init__(env)
        self.nTrajectories = numTrajectories
        self.nAgents = env.agents

        # Generate a new environment for each trajectory
        self.environments = [deepcopy(env) for _ in range(numTrajectories)]
        self.reset()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self):
        # self.states = []
        # for i in range(self.nTrajectories):
        #     self.env.reset()
        #     self.states.append(self.env.state)
        for env in self.environments:
            env.reset()

        self.doneIndices = []  # states are not indices here, thus we need a separate list
                               # to keep the indices the states that are done

    # Step given an nparray or list of actions
    # Input:
    #   actions: If scalar, apply to all trajectories
    #            If tuple, first item is action, second is trajectory index. Apply to one trajectory
    #            If nparray, apply to all states
    # Output:
    #   obs: nparray of observation indices for each trajectory. -1 for completed trajectories
    #   rewards: nparray of rewards for each trajectory
    #   done: nparray of whether a particular trajectory is done
    def step(self, actions):
        # actions - the actions to perform in the current time step across
        # all trajectories

        # Scalar action given, apply to all
        if np.isscalar(actions):
            actions = np.full(self.numTrajectories, actions, dtype=np.int32)

        # Make sure numpy array is of appropriate size
        assert actions.shape[0] == self.nTrajectories

        observations = []
        rewards = []
        dones = []

        for i in range(len(self.environments)):
            # For each simulation run that is done, return nothing
            if i in self.doneIndices:
                observations.append([-1] * self.nAgents)
                rewards.append(0)
                dones.append(True)
                continue

            env = self.environments[i]
            action = actions[i]
            obs, reward, done = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            if done:
                self.doneIndices.append(i)

        return np.array(observations), np.array(rewards), np.array(dones), {}

    # If multiprocessing, each worker will provide its trajectory index and desired action
    # def _stepForSingleWorker(self, action, trajectoryIndex):
    #     currState = self.state[trajectoryIndex]
    #
    #     # If this worker's episode is finished, return nothing
    #     if currState is None:
    #         return -1, 0.0, True, {}
    #
    #     newState = self.np_random.multinomial(1, self.env.T[currState, action]).argmax()
    #     obs = self.np_random.multinomial(1, self.env.O[currState, action, newState]).argmax()
    #     reward = self.env.R[currState, action, newState, obs]
    #     if self.env.episodic:
    #         done = self.env.D[currState, action]
    #     else:
    #         done = False
    #
    #     if done:
    #         self.state[trajectoryIndex] = -1
    #     else:
    #         self.state[trajectoryIndex] = newState
    #
    #     return obs, reward, done, {}
