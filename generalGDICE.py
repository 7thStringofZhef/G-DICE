import gym
import numpy as np
from multiprocessing import Pool
from GDICE_Python.Parameters import GDICEParams
from GDICE_Python.Controllers import FiniteStateControllerDistribution, DeterministicFiniteStateController
from GDICE_Python.Algorithms import runGDICEOnEnvironment
from GDICE_Python.Scripts import getGridSearchGDICEParams, saveResults, loadResults


def runBasic():
    envName = 'POMDP-4x3-episodic-v0'
    env = gym.make(envName)  # Make a gym environment with POMDP-1d-episodic-v0
    testParams = GDICEParams()  # Choose G-DICE parameters with default values
    controllerDistribution = FiniteStateControllerDistribution(testParams.numNodes, env.action_space.n, env.observation_space.n)  # make a controller with 10 nodes, with #actions and observations from environment
    #pool = Pool()  # Use a pool for parallel processing. Max # threads
    pool = None  # use a multiEnv for vectorized processing on computers with low memory or no core access

    # Run GDICE. Return the best average value, its standard deviation,
    # tables of the best deterministic transitions, and the updated distribution of controllers
    bestValue, bestValueStdDev, bestActionTransitions, bestNodeObservationTransitions, updatedControllerDistribution, \
    estimatedConvergenceIteration, allValues, allStdDev = \
        runGDICEOnEnvironment(env, controllerDistribution, testParams, parallel=pool)

    # Create a deterministic controller from the tables above
    bestDeterministicController = DeterministicFiniteStateController(bestActionTransitions, bestNodeObservationTransitions)

    # Test on environment

def runGridSearchOnOneEnv(envName):
    pool = Pool()
    GDICEList = getGridSearchGDICEParams()[1]
    env = gym.make(envName)
    for params in GDICEList:
        env.reset()
        FSCDist = FiniteStateControllerDistribution(params.numNodes, env.action_space.n, env.observation_space.n)
        results = runGDICEOnEnvironment(env, FSCDist, params, parallel=pool)
        saveResults('EndResults', envName, params, results)

def runGridSearchOnAllEnv():
    pool = Pool()
    envList, GDICEList = getGridSearchGDICEParams()
    for envStr in envList:
        try:
            env = gym.make(envStr)
        except MemoryError:
            print(envStr + ' too large for memory')
            continue
        for params in GDICEList:
            env.reset()
            FSCDist = FiniteStateControllerDistribution(params.numNodes, env.action_space.n, env.observation_space.n)
            try:
                results = runGDICEOnEnvironment(env, FSCDist, params, parallel=pool)
            except MemoryError:
                print(envStr + ' too large for parallel processing. Switching to MultiEnv...')
                results = runGDICEOnEnvironment(env, FSCDist, params, parallel=None)
            saveResults(envStr, params, results)






if __name__ == "__main__":
    # testres, testControllerDist, testParams = loadResults('GDICEResults/POMDP-hallway-episodic-v0/N5_K1000_S30_sim1000_B3_lr0.05_vTNone.npz')
    runGridSearchOnOneEnv('POMDP-hallway-episodic-v0')
