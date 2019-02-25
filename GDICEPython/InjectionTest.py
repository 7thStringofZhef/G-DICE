# -*- coding: utf-8 -*-

from GDICE_Python.Controllers import FiniteStateControllerDistribution
from GDICE_Python.Parameters import GDICEParams
from GDICE_Python.Algorithms import runGDICEOnEnvironment
from multiprocessing import Pool
import gym


if __name__ == "__main__":
    envName = 'DPOMDP-recycling-v0'
    env = gym.make(envName)
    testParams = GDICEParams([10, 10], centralized=False)
    controllers = [FiniteStateControllerDistribution(testParams.numNodes[a], env.action_space[a].n, env.observation_space[a].n, True) for a in range(env.agents)]
    pool = Pool()
    bestValue, bestValueStdDev, bestActionTransitions, bestNodeObservationTransitions, updatedControllerDistribution, estimatedConvergenceIteration, allValues, allStdDev, bestValueAtEachIteration, bestStdDevAtEachIteration = runGDICEOnEnvironment(env, controllers, testParams, parallel=pool)