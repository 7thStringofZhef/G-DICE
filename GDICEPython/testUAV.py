from GDICE_Python.Parameters import GDICEParams
from GDICE_Python.Controllers import FiniteStateControllerDistribution, DeterministicFiniteStateController
from GDICE_Python.Algorithms import runGDICEOnEnvironment
from GDICE_Python.Evaluation import runDeterministicControllerOnEnvironment
from GDICE_Python.UAVDomains import UAVSimpleDomain, UAVSingleAgentStaticTargetDomain

def runDomain(env):
    testParams = GDICEParams(numSamples=10,
                             numSimulationsPerSample=10,
                             numIterations=30)

    controllerDistribution = FiniteStateControllerDistribution(testParams.numNodes, env.action_space.n,
                                                               env.observation_space.n)  # make a controller with 10 nodes, with #actions and observations from environment

    # pool = Pool()  # Use a pool for parallel processing. Max # threads
    pool = None  # use a multiEnv for vectorized processing on computers with low memory or no core access

    # Run GDICE. Return the best average value, its standard deviation,
    # tables of the best deterministic transitions, and the updated distribution of controllers
    bestValue, bestValueStdDev, bestActionTransitions, bestNodeObservationTransitions, updatedControllerDistribution, \
    estimatedConvergenceIteration, allValues, allStdDev, bestValueAtEachIteration, bestStdDevAtEachIteration = \
        runGDICEOnEnvironment(env, controllerDistribution, testParams, parallel=pool, envType=1)

    # Create a deterministic controller from the tables above
    bestDeterministicController = DeterministicFiniteStateController(bestActionTransitions,
                                                                     bestNodeObservationTransitions)

    # Test on environment
    finalValue = runDeterministicControllerOnEnvironment(env, bestDeterministicController,
                                                         testParams.timeHorizon, printMsgs=True)
    print('Final value:', finalValue)



if __name__ == "__main__":
    #runDomain(UAVSimpleDomain())
    runDomain(UAVSingleAgentStaticTargetDomain(seed=0, n_rows=5, n_columns=5))