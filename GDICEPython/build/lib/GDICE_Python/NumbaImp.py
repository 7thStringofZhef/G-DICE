from numba import cuda, vectorize, njit, UniTuple, int32, float32, float64, prange, guvectorize
from copy import deepcopy
import numpy as np

"""
So how do I do this?

What parallelizes well?
-Number of simulations per sample: Maintain a state (int32) for each sim
-Number of samples: Get a value and stddev back for each sample
--If I really wanted to be clever, take the max # samples, clip off for each GDICE permutation
-Packing in the controller distribution would be cool too (get new samples)


Gym (DPOMDP):
  action_space: Won't matter. Implicit in FSC
  observation_space: Same as above
  np_random: Environment seed
  discount: scalar
  episodic: False
  O: 
  R:
  T:
  agents
  
Gym (POMDP):
  O: nStates*nActions*nStates*nObs float64
  R: nStates*nActions*nStates*nObs float64
  T: nStates*nActions*nStates float64
  state: 1
  
For multi gym:
state: (nTrajectories, ) int64
nTrajectories: duh

  
Gym can probably just be float32

"""


# Return T, O, R, state, random seed
def extractGymFieldsPOMDP(pomdp):
    return pomdp.T.astype(np.float32), pomdp.O.astype(np.float32), pomdp.R.astype(np.float32)



@njit(int32(UniTuple(float32[:], 2)(int32, float32[:, :, :], float32[:, :, :, :], float32[:, :, :, :], float32, int32[:, :], int32[:, :, :], int32, int32), parallel=True, nogil=True) parallel=True, nogil=True)
def njit_evalAllSamples(state, T, O, R, gamma, aTrans, ntTrans, horizon=100, nSim=1000):
    nSamples = aTrans.shape[-1]
    nAgents = aTrans.shape[-2]
    values = np.zeros(nSamples, dtype=np.float32)
    stdDevs = np.zeros(nSamples, dtype=np.float32)
    for sample in prange(aTrans.shape[-1]):
        val, stddev = njit_evalSingleSample(state, T, O, R, gamma, aTrans[:, sample], ntTrans[:, :, sample], horizon, nSim)
        values[sample] = val
        stdDevs[sample] = stddev
    return values, stdDevs

@njit(int32(UniTuple(float32[:], 2)(int32, float32[:, :, :], float32[:, :, :, :], float32[:, :, :, :], float32, int32[:, :], int32[:, :, :], int32, int32), parallel=True, nogil=True) parallel=True, nogil=True)
def njit_evalAllSamplesDPOMDP(state, T, O, R, gamma, aTrans, ntTrans, horizon=100, nSim=1000):
    nSamples = aTrans.shape[-1]
    values = np.zeros(nSamples, dtype=np.float32)
    stdDevs = np.zeros(nSamples, dtype=np.float32)
    for sample in prange(aTrans.shape[-1]):
        val, stddev = njit_evalSingleSample(state, T, O, R, gamma, aTrans[:, sample], ntTrans[:, :, sample], horizon, nSim)
        values[sample] = val
        stdDevs[sample] = stddev
    return values, stdDevs

@guvectorize(int32(UniTuple(float32, 2)(int32, float32[:, :, :], float32[:, :, :, :], float32[:, :, :, :], float32, int32[:, :], int32[:, :, :], int32, int32), parallel=True, nogil=True) parallel=True, nogil=True)
def guVec_evalAllSamples(state, T, O, R, gamma, aTrans, ntTrans, horizon=100, nSim=1000):
    pass

# Inputs:
#   state (int32)
#   transition matrix  (float32)
#   observation matrix  (float32)
#   reward matrix  (float32)
#   gamma (float32)
#   actionTransitions for this sample (int32)
#   nodeObsTransitions for this sample (int32)
#   horizon
#   nSim
# Outputs:
#   meanValue
#   standardDeviation
@njit(UniTuple(float32, 2)(int32, float32[:, :, :], float32[:, :, :, :], float32[:, :, :, :], float32, int32[:], int32[:, :], int32, int32),
      parallel=True, nogil=True)
def njit_evalSingleSample(state, T, O, R, gamma, aTrans, ntTrans, horizon=100, nSim=1000):
    currentNode = 0
    values = np.zeros(nSim, dtype=np.float32)
    for sim in prange(nSim):  # Can parallelize across sims
        for t in range(horizon):
            action = aTrans[currentNode]
            newState = np.random.multinomial(1, T[state, action]).argmax()
            obs = O[state, action, newState]
            values[sim] += (gamma ** t) * R[state, action, newState, obs]
            currentNode = ntTrans[obs, currentNode]
            state = newState
    return values.mean(), values.std()

@njit(UniTuple(float32, 2)(int32, float32[:, :, :, :], float32[:, :, :, :, :, :], float32[:, :, :, :, :, :], float32, int32[:, :], int32[:, :, :], int32, int32),
      parallel=True, nogil=True)
def njit_evalSingleSampleDPOMDP(state, T, O, R, gamma, aTrans, ntTrans, horizon=100, nSim=1000):
    nAgents = 2  # For now, just 2-agent DPOMDPs
    nObs = O.shape[-1]
    agentIndices = np.arange(nAgents, dtype=np.int32)
    currentNode = np.zeros(nAgents, dtype=np.int32)
    values = np.zeros(nSim, dtype=np.float32)
    for sim in prange(nSim):  # Can parallelize across sims
        for t in range(horizon):
            action = aTrans[currentNode, agentIndices]
            newState = np.random.multinomial(1, T[state, (*action)]).argmax().item()
            obs = np.unravel_index(np.random.multinomial(1, O[state, (*action), newState].flatten()).argmax(), tuple([nObs for _ in range(nAgents)]))
            values[sim] += (gamma ** t) * R[state, (*action), newState, (*obs)].item()
            currentNode = ntTrans[obs, currentNode, agentIndices]
            state = newState
    return values.mean(), values.std()


"""
def numba_evaluateSamplesPOMDP(env, numSimulationsPerSample, timeHorizon, actionTransitions, nodeObservationTransitions):
    numSamples = actionTransitions.shape[-1]  # Last entry
    gamma = env.discount if env.discount is not None else 1
    # Start state should be starting state (for each sample, for each simulation per sample)
    # nSamples*nSims
    if env.start is None:
        startState = np.array([env.np_random.randint(env.state_space.n, size=numSimulationsPerSample).astype(np.int32) for _ in range(numSamples)])
    else:
        startState = np.array([env.np_random.multinomial(1, env.start / np.sum(env.start),
                                                         size=numSimulationsPerSample).argmax(1).astype(np.int32) for _ in range(numSamples)])
    T, O, R = extractGymFieldsPOMDP(env)  # Extract needed matrices

    # Transfer environment to device
    envT = cuda.to_device(T)
    envO = cuda.to_device(O)
    envR = cuda.to_device(R)
    envStart = cuda.to_device(startState)
    envGamma = cuda.to_device(gamma)
    envHorizon = cuda.to_device(timeHorizon)

    # Transfer value and StdDev storage to device
    values = cuda.to_device(np.zeros(numSamples, dtype=np.float64))
    stdDevs = cuda.to_device(np.zeros(numSamples, dtype=np.float64))

    # Transfer samples
    aSamples = cuda.to_device(actionTransitions.astype(np.int32))
    ntSamples = cuda.to_device(nodeObservationTransitions.astype(np.int32))

    # Define cuda parameters

    # Run cuda eval
    cudaEval(values, stdDevs, envStart, envT, envO, envR, envGamma, envHorizon, aSamples, ntSamples)


# Values and stddevs are return values (numSamples,) float64
# startState is (numSamples, numSims) int32
# Max 1024 threads per block
@cuda.jit
def cudaEval(values, stdDevs, envStart, envT, envO, envR, envGamma, envHorizon, aSamples, ntSamples):
    pass


# Evaluate multiple trajectories for a sample, starting from first node
# Inputs:
#   env: MultiPOMDP environment in which to evaluate (numTrajectories is number of simulations)
#   timeHorizon: Time horizon over which to evaluate
#   actionTransitions: (numNodes,) int array of chosen actions for each node
#   nodeObservationTransitions: (numObs, numNodes) int array of chosen node transitions for obs
#  Output:
#    value: Discounted total return over timeHorizon (or until episode is done), averaged over all simulations
#    stdDev: Standard deviation of discounter total returns over all simulations
def evaluateSampleMultiPOMDP(env, timeHorizon, actionTransitions, nodeObservationTransitions):
    numTrajectories = env.nTrajectories
    gamma = env.discount if env.discount is not None else 1
    env.reset()
    currentNodes = np.zeros(numTrajectories, dtype=np.int32)
    currentTimestep = 0
    values = np.zeros(numTrajectories, dtype=np.float64)
    isDones = np.zeros(numTrajectories, dtype=bool)
    while not all(isDones) and currentTimestep < timeHorizon:
        obs, rewards, isDones = env.step(actionTransitions[currentNodes])[:3]
        currentNodes = nodeObservationTransitions[obs, currentNodes]
        values += rewards * (gamma ** currentTimestep)
        currentTimestep += 1

    return values.mean(axis=0), values.std(axis=0)
"""