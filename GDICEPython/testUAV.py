from GDICEPython.GDICE_Python import GDICEParams
from GDICEPython.GDICE_Python import FiniteStateControllerDistribution, DeterministicFiniteStateController
from GDICEPython.GDICE_Python import runGDICEOnEnvironment
from GDICEPython.GDICE_Python import UAVSimpleDomain

def runSimpleDomain():
    env = UAVSimpleDomain()
    testParams = GDICEParams(numSamples=10,
                             numSimulationsPerSample=10)

    controllerDistribution = FiniteStateControllerDistribution(testParams.numNodes, env.action_space.n,
                                                               env.observation_space.n)  # make a controller with 10 nodes, with #actions and observations from environment

    # pool = Pool()  # Use a pool for parallel processing. Max # threads
    pool = None  # use a multiEnv for vectorized processing on computers with low memory or no core access

    # Run GDICE. Return the best average value, its standard deviation,
    # tables of the best deterministic transitions, and the updated distribution of controllers
    bestValue, bestValueStdDev, bestActionTransitions, bestNodeObservationTransitions, updatedControllerDistribution, \
    estimatedConvergenceIteration, allValues, allStdDev = \
        runGDICEOnEnvironment(env, controllerDistribution, testParams, parallel=pool, envType=1)

    # Create a deterministic controller from the tables above
    bestDeterministicController = DeterministicFiniteStateController(bestActionTransitions,
                                                                     bestNodeObservationTransitions)

    # Test on environment

if __name__ == "__main__":
    runSimpleDomain()