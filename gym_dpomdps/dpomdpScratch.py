import sys
import numpy as np
from gym_dpomdps import list_dpomdps, MultiDPOMDP, DPOMDP
import os
import pickle
from GDICE_Python.Controllers import FiniteStateControllerDistribution
from GDICE_Python.Scripts import saveResults, loadResults, getGridSearchGDICEParamsDPOMDP, checkIfFinished, checkIfPartial
from GDICE_Python.Algorithms import runGDICEOnEnvironment
from GDICE_Python.Parameters import GDICEParams
from multiprocessing import Pool
import gym
import glob

def runBasicDPOMDP():
    envName = 'DPOMDP-dectiger-v0'
    env = gym.make(envName)
    testParams = GDICEParams([10, 10])
    controllers = [FiniteStateControllerDistribution(testParams.numNodes[a], env.action_space[a].n, env.observation_space[a].n) for a in range(env.agents)]
    pool = Pool()
    bestValue, bestValueStdDev, bestActionTransitions, bestNodeObservationTransitions, updatedControllerDistribution, \
    estimatedConvergenceIteration, allValues, allStdDev, bestValueAtEachIteration, bestStdDevAtEachIteration = \
        runGDICEOnEnvironment(env, controllers, testParams, parallel=pool)

def runGridSearchOnOneEnvDPOMDP(baseSavePath, envName):
    pool = Pool()
    GDICEList = getGridSearchGDICEParamsDPOMDP()[1]
    try:
        env = gym.make(envName)
    except MemoryError:
        print(envName + ' too large for memory', file=sys.stderr)
        return
    except Exception as e:
        print(envName + ' encountered error in creation', file=sys.stderr)
        print(e, file=sys.stderr)
        return

    for params in GDICEList:
        # Skip this permutation if we already have final results
        if checkIfFinished(envName, params.name, baseDir=baseSavePath)[0]:
            print(params.name + ' already finished for ' + envName + ', skipping...', file=sys.stderr)
            continue
        wasPartiallyRun, npzFilename = checkIfPartial(envName, params.name)
        prevResults = None
        if wasPartiallyRun:
            print(params.name + ' partially finished for ' + envName + ', loading...', file=sys.stderr)
            prevResults, FSCDist = loadResults(npzFilename)[:2]
        else:
            FSCDist = [FiniteStateControllerDistribution(params.numNodes[a], env.action_space[a].n,
                                                        env.observation_space[a].n) for a in range(env.agents)]
        env.reset()
        try:
            results = runGDICEOnEnvironment(env, FSCDist, params, parallel=pool, results=prevResults, baseDir=baseSavePath)
        except MemoryError:
            print(envName + ' too large for parallel processing. Switching to MultiEnv...', file=sys.stderr)
            results = runGDICEOnEnvironment(env, FSCDist, params, parallel=None, results=prevResults, baseDir=baseSavePath)
        except Exception as e:
            print(envName + ' encountered error in runnning' + params.name + ', skipping to next param', file=sys.stderr)
            print(e, file=sys.stderr)
            continue
        saveResults(os.path.join(baseSavePath, 'EndResults'), envName, params, results)
        # Delete the temp results
        try:
            for filename in glob.glob(os.path.join(baseSavePath, 'GDICEResults', envName, params.name)+'*'):
                os.remove(filename)
        except:
            continue


# Run a grid search on all registered environments
def runGridSearchOnAllEnvDPOMDP(baseSavePath):
    pool = Pool()
    envList, GDICEList = getGridSearchGDICEParamsDPOMDP()
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
            if checkIfFinished(envStr, params.name, baseDir=baseSavePath)[0]:
                print(params.name +' already finished for ' +envStr+ ', skipping...', file=sys.stderr)
                continue

            wasPartiallyRun, npzFilename = checkIfPartial(envStr, params.name)
            prevResults = None
            if wasPartiallyRun:
                print(params.name + ' partially finished for ' + envStr + ', loading...', file=sys.stderr)
                prevResults, FSCDist = loadResults(npzFilename)[:2]
            else:
                FSCDist = [FiniteStateControllerDistribution(params.numNodes[a], env.action_space[a].n,
                                                             env.observation_space[a].n) for a in range(env.agents)]
            env.reset()
            try:
                results = runGDICEOnEnvironment(env, FSCDist, params, parallel=pool, results=prevResults, baseDir=baseSavePath)
            except MemoryError:
                print(envStr + ' too large for parallel processing. Switching to MultiEnv...', file=sys.stderr)
                results = runGDICEOnEnvironment(env, FSCDist, params, parallel=None, results=prevResults, baseDir=baseSavePath)
            except Exception as e:
                print(envStr + ' encountered error in runnning' + params.name + ', skipping to next param', file=sys.stderr)
                print(e, file=sys.stderr)
                continue

            saveResults(os.path.join(baseSavePath, 'EndResults'), envStr, params, results)
            # Delete the temp results
            try:
                for filename in glob.glob(os.path.join(baseSavePath, 'GDICEResults', envStr, params.name) + '*'):
                    os.remove(filename)
            except:
                continue

if __name__=="__main__":
    baseSavePath = ''
    runGridSearchOnAllEnvDPOMDP(baseSavePath)
    #runBasicDPOMDP()
