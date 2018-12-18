#import sys
#sys.path.append('gym_dpomdps/')
import numpy as np
from gym_dpomdps import list_dpomdps, MultiDPOMDP
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
#   env: MultiEnv environment in which to evaluate
#   timeHorizon: Time horizon over which to evaluate
#   actionTransitions: (numNodes,) int array of chosen actions for each node
#   nodeObservationTransitions: (numObs, numNodes) int array of chosen node transitions for obs
#  Output:
#    allSampleValues: Discounted total return over timeHorizon (or until episode is done), averaged over all simulations, for each sample (numSamples,)
#    stdDevs: Standard deviation of discounted total returns over all simulations, for each sample (numSamples,)
def evaluateSamplesMultiEnv(env, timeHorizon, actionTransitions, nodeObservationTransitions):
    assert isinstance(env, MultiDPOMDP)
    nTrajectories = env.nTrajectories
    nAgents = env.agents
    gamma = env.discount if env.discount is not None else 1
    env.reset()
    currentNodes = [np.zeros(nTrajectories, dtype=np.int32) for _ in range(nAgents)]  # 50*2
    currentTimestep = 0
    values = np.zeros(nTrajectories, dtype=np.float64)
    isDones = np.zeros(nTrajectories, dtype=bool)
    while not all(isDones) and currentTimestep < timeHorizon:
        jointActions = np.stack([actionTransitions[currentNodes[i], i] for i in range(nAgents)], axis=1)
        obs, rewards, isDones = env.step(jointActions)[:3]  # Want a 50*2
        currentNodes = [nodeObservationTransitions[obs[:,i], currentNodes[i], i] for i in range(nAgents)]
        values += rewards * (gamma ** currentTimestep)
        currentTimestep += 1

    return values.mean(axis=0), values.std(axis=0)

if __name__=="__main__":
    env = gym.make(list_dpomdps()[1])
    multiEnv = MultiDPOMDP(env, 50)
    controller1 = FiniteStateControllerDistribution(10, env.action_space[0].n, env.observation_space[0].n)
    controller2 = FiniteStateControllerDistribution(10, env.action_space[1].n, env.observation_space[1].n)
    controllers = [controller1, controller2]

    sampledActions = np.stack([controller.sampleActionFromAllNodes(50) for controller in controllers], axis=-1)  # numNodes*numSamples*numAgents
    sampledNodes = np.stack([controller.sampleAllObservationTransitionsFromAllNodes(50) for controller in controllers], axis=-1)  # numObs*numBeginNodes*numSamples*numAgents
    iterActions = sampledActions[:, 0, :]  # numNodes*numAgents (10*2)
    iterObs = sampledNodes[:, :, 0, :]  # numobs*numStartNodes*numAgents
    test = evaluateSamplesMultiEnv(multiEnv, 50, iterActions, iterObs)
    pass

