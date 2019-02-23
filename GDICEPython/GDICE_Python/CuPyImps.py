import cupy as cp
import numpy as np

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

# Transfer environment to gpu first
def sendGymToGPU(pomdp):
    T, O, R, nAgents, nActions, nObs, nStates, discount, start = extractGymFieldsPOMDP(pomdp)
    T = cp.asarray(T)  # Transition matrix
    O = cp.asarray(O)  # Observation transition matrix
    R = cp.asarray(R)  # Reward matrix
    start = cp.asarray(start)  # Start probability (for reset)
    return T, O, R, start