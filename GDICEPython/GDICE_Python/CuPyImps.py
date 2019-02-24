import cupy as cp
import cupy.random as cpr
import numpy as np
import numpy.random as npr

from gym_dpomdps import list_dpomdps
from gym_pomdps import list_pomdps
import gym

from GDICE_Python.Controllers import FiniteStateControllerDistribution
from GDICE_Python.Utils import sampleFromControllerDistribution

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
    #T, O = normalizeProbTablesPOMDP(T, O)

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
    T = cp.array(T)  # Transition matrix
    O = cp.array(O)  # Observation transition matrix
    R = cp.array(R)  # Reward matrix
    start = cp.array(start)  # Start probability (for reset)
    return T, O, R, start

# Init stuff on GPU
def initOnGPU(aTrans, ntTrans, startG, nSim=1000):
    nN = aTrans.shape[0]
    nSamples = aTrans.shape[1]

    # Reset state and nodes
    stateG = cp.zeros((nSamples, nSim), dtype=cp.int32)
    currNodeG = cp.zeros((nSamples, nSim), dtype=cp.int32)
    stateG = resetGymGPU(stateG, currNodeG, startG)

    # Send new set of sampled transitions
    aTransG = cp.array(aTrans)  # nN*nSamples
    ntTransG = cp.array(ntTrans)  # nO*nN*nSamples

    # Create things to store rewards, obs
    newStateG = cp.zeros((nSamples, nSim), dtype=cp.int32)
    obsG = cp.zeros((nSamples, nSim), dtype=cp.int32)
    valueG = cp.zeros((nSamples, nSim), dtype=cp.float64)

    # Convenience indices
    sampIndG = cp.arange(nSamples)  # 0:50
    nodeIndG = cp.arange(nN)

    return stateG, currNodeG, aTransG, ntTransG, newStateG, obsG, valueG, sampIndG, nodeIndG


# Reset state to start
def resetGymGPU(stateG, currNodeG, startG):
    stateG = cpr.multinomial(1, startG, size=(50,1000)).argmax(2)
    currNodeG.fill(0)
    return stateG

def resetIntermediateGPU(valueG):
    valueG.fill(0.0)

def stepGPU(stateG, newStateG, tG, oG, rG, currNodeG, aTransG, ntTransG, gamma, sampIndG, nodeIndG):
    nSim = stateG.shape[-1]
    nSamp = stateG.shape[0]
    fullSampIndices = cp.tile(cp.expand_dims(sampIndG, 1), (1, nSim))
    actionsG = aTransG[currNodeG, fullSampIndices]  #50*1000 actions at this timestep for each sample-sim combination
    #newStateG = cpr.multinomial(1, tG[s, a]) for s
    test = tG[stateG, actionsG]
    pass
    #np.array([cpr.multinomial(1, p).argmax() for p in tG[stateG, aTransG]])



if __name__ == "__main__":
    env = gym.make('POMDP-4x3-episodic-v0')
    tH = 100
    controllerDistribution = FiniteStateControllerDistribution(10, env.action_space.n, env.observation_space.n)
    tG, oG, rG, startG = sendGymToGPU(env)
    aTrans, ntTrans = sampleFromControllerDistribution(controllerDistribution, 50)
    stateG, currNodeG, aTransG, ntTransG, newStateG, obsG, valueG, sampIndG, nodeIndG = initOnGPU(aTrans, ntTrans, startG)
    stepGPU(stateG, newStateG, tG, oG, rG, currNodeG, aTransG, ntTransG, 0.99, sampIndG, nodeIndG)
    for t in range(tH):

        pass
