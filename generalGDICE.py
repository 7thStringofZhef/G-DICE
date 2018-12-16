import gym
import os
import argparse
import sys
from multiprocessing import Pool
from GDICE_Python.Parameters import GDICEParams
from GDICE_Python.Controllers import FiniteStateControllerDistribution, DeterministicFiniteStateController
from GDICE_Python.Algorithms import runGDICEOnEnvironment
from GDICE_Python.Scripts import getGridSearchGDICEParams, saveResults, loadResults, checkIfFinished, checkIfPartial


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

def runGridSearchOnOneEnv(baseSavePath, envName):
    pool = Pool()
    GDICEList = getGridSearchGDICEParams()[1]
    env = gym.make(envName)
    for params in GDICEList:
        env.reset()
        FSCDist = FiniteStateControllerDistribution(params.numNodes, env.action_space.n, env.observation_space.n)
        results = runGDICEOnEnvironment(env, FSCDist, params, parallel=pool)
        saveResults(os.path.join(baseSavePath, 'EndResults'), envName, params, results)


# Run a grid search on all registered environments
def runGridSearchOnAllEnv(baseSavePath):
    pool = Pool()
    envList, GDICEList = getGridSearchGDICEParams()
    for envStr in envList:
        try:
            env = gym.make(envStr)
        except MemoryError:
            print(envStr + ' too large for memory', file=sys.stderr)
            continue
        except Exception as e:
            print(envStr + ' encountered error in creation, skipping', file=sys.stderr)
            print(e, file=sys.stderr)
            continue
        for params in GDICEList:
            # Skip this permutation if we already have final results
            if checkIfFinished(envStr, params.name)[0]:
                print(params.name +' already finished for ' +envStr+ ', skipping...', file=sys.stderr)
                continue

            wasPartiallyRun, npzFilename = checkIfPartial(envStr, params.name)
            prevResults = None
            if wasPartiallyRun:
                print(params.name + ' partially finished for ' + envStr + ', loading...', file=sys.stderr)
                prevResults, FSCDist = loadResults(npzFilename)[:2]
            else:
                FSCDist = FiniteStateControllerDistribution(params.numNodes, env.action_space.n,
                                                            env.observation_space.n)
            env.reset()
            try:
                results = runGDICEOnEnvironment(env, FSCDist, params, parallel=pool, results=prevResults)
            except MemoryError:
                print(envStr + ' too large for parallel processing. Switching to MultiEnv...', file=sys.stderr)
                results = runGDICEOnEnvironment(env, FSCDist, params, parallel=None, results=prevResults)
            except Exception as e:
                print(envStr + ' encountered error in runnning' + params.name + ', skipping to next param', file=sys.stderr)
                print(e, file=sys.stderr)
                continue

            saveResults(os.path.join(baseSavePath, 'EndResults'), envStr, params, results)


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description='Choose save dir and environment')
    parser.add_argument('--save_path', type=str, nargs=1,
                        help='End result save path')
    parser.add_argument('--env-name', action='store_const', default='',
                        help='Environment to run')

    args = parser.parse_args()
    """
    #baseSavePath = ''
    baseSavePath = '/scratch/slayback.d/GDICE'
    # testres, testControllerDist, testParams = loadResults('GDICEResults/POMDP-hallway-episodic-v0/N5_K1000_S30_sim1000_B3_lr0.05_vTNone.npz')
    # runGridSearchOnOneEnv('POMDP-hallway-episodic-v0')
    runGridSearchOnAllEnv(baseSavePath)
