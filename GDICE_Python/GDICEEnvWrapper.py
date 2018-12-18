import gym
import numpy as np

# A class that wraps a gym environment in which states, observations, rewards, actions and dones
# are lists or np arrays, so multiple trajectories can be simulated in parallel
class GDICEEnvWrapper(gym.Wrapper):
    def __init__(self, env, numTrajectories):
        super().__init__(env)
        self.numTrajectories = numTrajectories
        self.reset()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self):
        self.env.reset()
        self.states = [self.env.state] * self.numTrajectories
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

        # Tuple of (action, index) given, step for one worker only (for multithreaded applications)
        #if isinstance(actions, tuple) and len(actions) == 2:
        #    return self._stepForSingleWorker(int(actions[0]), int(actions[1]))

        # Make sure numpy array is of appropriate size
        assert actions.shape[0] == self.numTrajectories

        newStates = []
        observations = []
        rewards = []
        dones = []

        for i in range(len(self.states)):
            # For each agent that is done, return nothing
            if i in self.doneIndices:
                newStates.append(-1)
                observations.append(-1)
                rewards.append(0)
                dones.append(True)
                continue

            state = self.states[i]
            action = actions[i]

            obs, reward, done, params = self.env.step(action, state)

            newState = params['new_state']
            newStates.append(newState)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            if done:
                self.doneIndices.append(i)

        self.states = newStates
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
