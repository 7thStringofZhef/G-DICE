import gym
import os
import argparse
import sys
from gym_dpomdps import list_dpomdps
from gym_pomdps import list_pomdps
from multiprocessing import Pool
from GDICE_Python.Parameters import GDICEParams
from GDICE_Python.Controllers import FiniteStateControllerDistribution, DeterministicFiniteStateController
from GDICE_Python.Algorithms import runGDICEOnEnvironment
from GDICE_Python.Scripts import getGridSearchGDICEParams, saveResults, loadResults, checkIfFinished, checkIfPartial, claimRunEnvParamSet, registerRunEnvParamSetCompletion, claimRunEnvParamSet_unfinished, registerRunEnvParamSetCompletion_unfinished
import glob

def runBasicDPOMDP():
    envName = 'DPOMDP-dectiger-v0'
    env = gym.make(envName)
    testParams = GDICEParams(10, centralized=False)
    controllers = [FiniteStateControllerDistribution(testParams.numNodes, env.action_space[a].n, env.observation_space[a].n) for a in range(env.agents)]
    pool = Pool()
    bestValue, bestValueStdDev, bestActionTransitions, bestNodeObservationTransitions, updatedControllerDistribution, \
    estimatedConvergenceIteration, allValues, allStdDev, bestValueAtEachIteration, bestStdDevAtEachIteration = \
        runGDICEOnEnvironment(env, controllers, testParams, parallel=pool)

if __name__ == "__main__":
    runBasicDPOMDP()

    pass