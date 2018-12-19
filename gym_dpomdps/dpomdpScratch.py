#import sys
#sys.path.append('gym_dpomdps/')
import numpy as np
from gym_dpomdps import list_dpomdps, MultiDPOMDP, DPOMDP
import os
import pickle
from GDICE_Python.Controllers import FiniteStateControllerDistribution
import gym

# Save the results of a run
def saveResults(baseDir, envName, testParams, results):
    print('Saving...')
    savePath = os.path.join(baseDir, 'GDICEResults', envName)  # relative to current path
    os.makedirs(savePath, exist_ok=True)
    bestValue, bestValueStdDev, bestActionTransitions, bestNodeObservationTransitions, updatedControllerDistribution, \
    estimatedConvergenceIteration, allValues, allStdDev, bestValueAtEachIteration, bestStdDevAtEachIteration = results
    np.savez(os.path.join(savePath, testParams.name)+'.npz', bestValue=bestValue, bestValueStdDev=bestValueStdDev,
             bestActionTransitions=bestActionTransitions, bestNodeObservationTransitions=bestNodeObservationTransitions,
             estimatedConvergenceIteration=estimatedConvergenceIteration, allValues=allValues, allStdDev=allStdDev,
             bestValueAtEachIteration=bestValueAtEachIteration, bestStdDevAtEachIteration=bestStdDevAtEachIteration)
    pickle.dump(updatedControllerDistribution, open(os.path.join(savePath, testParams.name)+'.pkl', 'wb'))
    pickle.dump(testParams, open(os.path.join(savePath, testParams.name+'_params') + '.pkl', 'wb'))


# Load the results of a run
# Inputs:
#   filePath: Path to any of the files.
# Outputs:
#   All the results generated
#   The updated controller distribution
#   The GDICE params object
def loadResults(filePath):
    baseName = os.path.splitext(filePath)[0]
    print('Loading...')
    fileDict = np.load(baseName+'.npz')
    keys = ('bestValue', 'bestValueStdDev', 'bestActionTransitions', 'bestNodeObservationTransitions',
            'estimatedConvergenceIteration', 'allValues', 'allStdDev', 'bestValueAtEachIteration',
            'bestStdDevAtEachIteration')
    results = tuple([fileDict[key] for key in keys])
    updatedControllerDistribution = pickle.load(open(baseName+'.pkl', 'rb'))
    params = pickle.load(open(baseName + '_params.pkl', 'rb'))
    return results, updatedControllerDistribution, params

def _initGDICERunVariables(params):
    # Start variables
    bestActionProbs = None
    bestNodeTransitionProbs = None
    bestValue = np.NINF
    bestValueAtEachIteration = np.full(params.numIterations, np.nan, dtype=np.float64)
    bestStdDevAtEachIteration = np.full(params.numIterations, np.nan, dtype=np.float64)
    bestValueVariance = 0
    worstValueOfPreviousIteration = np.NINF
    allValues = np.zeros((params.numIterations, params.numSamples), dtype=np.float64)
    allStdDev = np.zeros((params.numIterations, params.numSamples), dtype=np.float64)
    estimatedConvergenceIteration = 0
    startIter = 0
    return bestValue, bestValueVariance, bestActionProbs, bestNodeTransitionProbs, estimatedConvergenceIteration, \
    allValues, allStdDev, bestValueAtEachIteration, bestStdDevAtEachIteration, startIter, worstValueOfPreviousIteration


# Evaluate multiple trajectories for a sample, starting from first node
# Inputs:
#   env: MultiDPOMDP environment in which to evaluate (numTrajectories is number of simulations)
#   timeHorizon: Time horizon over which to evaluate
#   actionTransitions: (numNodes, numAgents) int array of chosen actions for each node
#   nodeObservationTransitions: (numObs, numNodes, numAgents) int array of chosen node transitions for obs
#  Output:
#    value: Discounted total return over timeHorizon (or until episode is done), averaged over all simulations
#    stdDev: Standard deviation of discounter total returns over all simulations

def evaluateSampleMultiDPOMDP(env, timeHorizon, actionTransitions, nodeObservationTransitions):
    assert isinstance(env, MultiDPOMDP)
    nTrajectories = env.nTrajectories
    nAgents = env.agents
    agentIndices = tuple(np.full(nTrajectories, a, dtype=np.int32) for a in range(nAgents))
    gamma = env.discount if env.discount is not None else 1
    env.reset()
    currentNodes = tuple(np.zeros(nTrajectories, dtype=np.int32) for _ in range(nAgents))
    currentTimestep = 0
    values = np.zeros(nTrajectories, dtype=np.float64)
    isDones = np.zeros(nTrajectories, dtype=bool)
    while not all(isDones) and currentTimestep < timeHorizon:
        obs, rewards, isDones = env.step(actionTransitions[currentNodes, agentIndices].T)[:3]
        currentNodes = nodeObservationTransitions[tuple(obs[:, i] for i in range(nAgents)), currentNodes, agentIndices]
        values += rewards * (gamma ** currentTimestep)
        currentTimestep += 1

    return values.mean(axis=0), values.std(axis=0)

# Evaluate a single sample, starting from first node
# Inputs:
#   env: DPOMDP environment in which to evaluate
#   timeHorizon: Time horizon over which to evaluate
#   actionTransitions: (numNodes, ) int array of chosen actions for each node
#   nodeObservationTransitions: (numObs, numNodes) int array of chosen node transitions for obs
#  Output:
#    value: Discounted total return over timeHorizon (or until episode is done), averaged over all simulations
#    stdDev: Standard deviation of discounter total returns over all simulations
def evaluateSampleDPOMDP(env, timeHorizon, actionTransitions, nodeObservationTransitions, numSimulations):
    assert isinstance(env, DPOMDP)
    nAgents = env.agents
    agentIndices = np.arange(nAgents, dtype=int)
    gamma = env.discount if env.discount is not None else 1
    values = np.zeros(numSimulations, dtype=np.float64)
    for sim in range(numSimulations):
        env.reset()
        currentNodeIndex = [0 for _ in range(nAgents)]
        currentTimestep = 0
        isDone = False
        value = 0.0
        while not isDone and currentTimestep < timeHorizon:
            obs, reward, isDone = env.step(actionTransitions[currentNodeIndex, agentIndices])[:3]
            currentNodeIndex = nodeObservationTransitions[obs, currentNodeIndex, agentIndices]
            value += reward * (gamma ** currentTimestep)
            currentTimestep += 1
        values[sim] = value
    return values.mean(), values.std()

if __name__=="__main__":
    env = gym.make(list_dpomdps()[2])
    multiEnv = MultiDPOMDP(env, 50)
    controller1 = FiniteStateControllerDistribution(10, env.action_space[0].n, env.observation_space[0].n)
    controller2 = FiniteStateControllerDistribution(10, env.action_space[1].n, env.observation_space[1].n)
    controllers = [controller1, controller2]

    sampledActions = np.stack([controller.sampleActionFromAllNodes(50) for controller in controllers], axis=-1)  # numNodes*numSamples*numAgents
    sampledNodes = np.stack([controller.sampleAllObservationTransitionsFromAllNodes(50) for controller in controllers], axis=-1)  # numObs*numBeginNodes*numSamples*numAgents
    iterActions = sampledActions[:, 0, :]  # numNodes*numAgents (10*2)
    iterObs = sampledNodes[:, :, 0, :]  # numobs*numStartNodes*numAgents
    #test1 = evaluateSampleDPOMDP(env, 50, iterActions, iterObs, 1)
    test = evaluateSampleMultiDPOMDP(multiEnv, 50, iterActions, iterObs)
    pass

