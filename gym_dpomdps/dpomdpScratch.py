#import sys
#sys.path.append('gym_dpomdps/')
import numpy as np
from gym_dpomdps import list_dpomdps, MultiDPOMDP, DPOMDP
import os
import pickle
from GDICE_Python.Controllers import FiniteStateControllerDistribution
from GDICE_Python.Scripts import saveResults, loadResults
from GDICE_Python.Evaluation import evaluateSampleDPOMDP, evaluateSampleMultiDPOMDP
from GDICE_Python.Utils import _checkControllerDist, _checkEnv
from GDICE_Python.Algorithms import runGDICEOnEnvironment
from GDICE_Python.Parameters import GDICEParams
from multiprocessing import Pool
import gym

def runBasicDPOMDP():
    envName = 'DPOMDP-dectiger-v0'
    env = gym.make(envName)
    testParams = GDICEParams([10, 10])
    controllers = [FiniteStateControllerDistribution(testParams.numNodes[a], env.action_space[a].n, env.observation_space[a].n) for a in range(env.agents)]
    pool = Pool()
    bestValue, bestValueStdDev, bestActionTransitions, bestNodeObservationTransitions, updatedControllerDistribution, \
    estimatedConvergenceIteration, allValues, allStdDev, bestValueAtEachIteration, bestStdDevAtEachIteration = \
        runGDICEOnEnvironment(env, controllers, testParams, parallel=pool)

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

# Sample actions and node transitions from controller distribution(s)
# For each node in each controller, sample actions numNodes*numSamples*numControllers
# For each node, observation in controller, sample next node numObs*numBeginNodes*numSamples*numControllers
def _sampleFromControllerDistribution(controller, numSamples):
    if isinstance(controller, (list, tuple)):
        nC = len(controller)
        return np.stack([controller[a].sampleActionFromAllNodes(numSamples) for a in range(nC)], axis=-1), \
               np.stack([controller[a].sampleAllObservationTransitionsFromAllNodes(numSamples) for a in range(nC)], axis=-1)
    else:
        return controller.sampleActionFromAllNodes(numSamples), controller.sampleAllObservationTransitionsFromAllNodes(numSamples)

if __name__=="__main__":
    runBasicDPOMDP()
