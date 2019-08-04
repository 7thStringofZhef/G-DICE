import numpy as np
import numba as nb
from gym_dpomdps import list_dpomdps
from gym_pomdps import list_pomdps
from multiprocessing import Pool
from GDICE_Python.Parameters import GDICEParams
from GDICE_Python.Controllers import FiniteStateControllerDistribution, DeterministicFiniteStateController
from GDICE_Python.Algorithms import runGDICEOnEnvironment
import gym

# Extract gym fields so they can be used in a JIT function
# T, O, R, nAgents, nActions, nObs, nStates, discount, start
def extractGymFieldsPOMDP(pomdp):
    discount = pomdp.discount
    nStates = pomdp.state_space.n
    start = pomdp.start / np.sum(pomdp.start)
    if hasattr(pomdp, 'agents'):
        nAgents = pomdp.agents
        nActions = np.array([space.n for space in pomdp.action_space])  # nActions per agent
        nObs = np.array([space.n for space in pomdp.action_space])  # nObs per agent
    else:
        nAgents = 1
        nActions = pomdp.action_space.n
        nObs = pomdp.observation_space.n
    T, O, R = pomdp.T, pomdp.O, pomdp.R
    T, O = normalizeProbTablesPOMDP(T, O)

    return T, O, R, nAgents, nActions, nObs, nStates, discount, start

def normalizeProbTablesPOMDP(T, O):
    for s in range(T.shape[0]):
        for act in range(T.shape[1]):
            T[s, act, :] /= np.sum(T[s, act, :])
            for s2 in range(T.shape[2]):
                O[s,act,s2,:] /= np.sum(O[s,act,2,:])

    return T, O


# Return a set of start states (nTrajectories, ) of state index
def getStartState(start, nTrajectories=1000):
    return np.random.multinomial(1, start / np.sum(start), size=nTrajectories).argmax(1)

# Step in a POMDP (one trajectory)
@nb.njit(nb.types.Tuple((nb.int32, nb.float32, nb.int32))(nb.int32, nb.float32[:,:,:], nb.float32[:,:,:,:], nb.float32[:,:,:,:],nb.int32))
def step(action, T, O, R, state):
    state1 = np.random.multinomial(1, T[state, action]).argmax().item()
    obs = np.random.multinomial(1, O[state, action, state1]).argmax().item()
    reward = R[state, action, state1, obs].item()
    return obs, reward, state1

# Step in a DPOMDP (one trajectory)
@nb.njit(nb.types.Tuple((nb.int32[:], nb.float32, nb.int32))(nb.int32[:], nb.float32[:,:,:,:], nb.float32[:,:,:,:,:,:], nb.float32[:,:,:,:,:,:], nb.int32[:], nb.int32))
def stepDP(actions, T, O, R, nObs, state):
    state1 = np.int32(np.random.multinomial(1, T[state, actions[0], actions[1]]).argmax().item())
    obsI = np.random.multinomial(1, O[state, actions[0], actions[1], state1].flatten()).argmax()
    obs = np.int32((obsI//nObs[0], obsI % nObs[1]))
    reward = R[state, actions[0], actions[1], state1, obs[0], obs[1]].item()
    return obs, reward, state1


# Step in a POMDP (many trajectories)
@nb.njit(nb.types.Tuple((nb.int32[:], nb.float32[:]))(nb.int32[:], nb.float64[:,:,:], nb.float64[:,:,:,:], nb.float64[:,:,:,:], nb.int32[:]))
def stepMultiPOMDP(action, T, O, R, state):
    state1 = np.array([np.random.multinomial(1, T[state[i], action[i], :]).argmax().item() for i in range(state.shape[0])],dtype=np.int32)
    obs = np.array([np.random.multinomial(1, O[state[i], action[i], state1[i], :]).argmax().item() for i in range(state.shape[0])],dtype=np.int32)
    #reward = R[state, action, state1, obs]
    reward = np.zeros(state.shape[0], dtype=np.float32)
    for i in range(state.shape[0]):
        reward[i] = R[state[i], action[i], state1[i], obs[i]]
    state[:] = state1[:]
    return obs, reward


if __name__ == "__main__":
    envPO = gym.make('POMDP-4x3-episodic-v0')
    envDP = gym.make('DPOMDP-recycling-v0')
    poT, poO, poR = extractGymFieldsPOMDP(envPO)[:3]
    dpT, dpO, dpR = extractGymFieldsPOMDP(envPO)[:3]
    action = np.zeros(1000, np.int32)
    state = np.zeros(1000, np.int32)
    state1 = np.array([np.random.multinomial(1, poT[state[i], action[i], :]).argmax().item() for i in range(state.shape[0])],
                      dtype=np.int32)
    obs = np.array(
        [np.random.multinomial(1, poO[state[i], action[i], state1[i], :]).argmax().item() for i in range(state.shape[0])],
        dtype=np.int32)
    test = stepMultiPOMDP(action, poT, poO, poR, state)

    pass