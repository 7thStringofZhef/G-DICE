import numpy as np
from .Domains import MultiPOMDP
from gym_dpomdps import MultiDPOMDP
from .GDICEEnvWrapper import GDICEEnvWrapper
from .Scripts import saveResults
from .Evaluation import *
from .Utils import _initGDICERunVariables, _parsePartialResultsToGDICERunVariables, _checkEnv, _checkControllerDist, \
    sampleFromControllerDistribution, updateControllerDistribution

# Run GDICE with controller(s) on an environment, given
# Inputs:
#   env: Gym-like environment to evaluate on
#   controller: A controller or list of controllers corresponding to agents in the environment
#   params: GDICEParams object
#   timeHorizon: Number of timesteps to evaluate to. If None, run each sample until episode is finished
#   parallel: Attempt to use python multiprocessing across samples. If not None, should be a Pool object
#   convergenceThreshold: If set, attempts to detect early convergence within a run and stop before all iterations are done
#   saveFrequency: How frequently to save results in the middle of a run (numIterations between saves)
#   baseDir: Where to save temp results relative to. Defaults to current directory
#   envType: 0 if standard MultiPOMDP, anything else for GDICEEnvWrapper
def runGDICEOnEnvironment(env, controller, params, parallel=None, results=None, convergenceThreshold=0, saveFrequency=50, baseDir='', envType=0):
    nAgents, nActions, nObs = _checkEnv(env)
    nNodes, nActionsC, nObsC = _checkControllerDist(controller)
    # Ensure controller matches environment
    assert nActions == nActionsC and nObs == nObsC
    # Ensure params match controllers
    if isinstance(nNodes, int): assert nNodes == params.numNodes
    else: assert tuple(nNodes) == tuple(params.numNodes)

    # Choose appropriate evaluation function
    envEvalFn = evaluateSampleMultiDPOMDP if nAgents > 1 else evaluateSampleMultiPOMDP

    # Swap the wrapper function if using other type of environment
    MultiEnvWrapper = GDICEEnvWrapper if envType else MultiPOMDP if nAgents==1 else MultiDPOMDP

    timeHorizon = params.timeHorizon
    if results is None:  # Not continuing previous results
        # Reset controller
        if nAgents == 1: controller.reset()
        else: [c.reset() for c in controller]
        # Start variables
        bestValue, bestValueVariance, bestActionProbs, bestNodeTransitionProbs, estimatedConvergenceIteration, \
        allValues, allStdDev, bestValueAtEachIteration, bestStdDevAtEachIteration, startIter, \
        worstValueOfPreviousIteration = _initGDICERunVariables(params)
    else:  # Continuing
        bestValue, bestValueVariance, bestActionProbs, bestNodeTransitionProbs, estimatedConvergenceIteration, \
        allValues, allStdDev, bestValueAtEachIteration, bestStdDevAtEachIteration, startIter, \
        worstValueOfPreviousIteration =_parsePartialResultsToGDICERunVariables(params, results)


    iterBestValue = np.NINF  # What is the most recently seen best controller value
    for iteration in range(startIter, params.numIterations):
        sampledActions, sampledNodes = sampleFromControllerDistribution(controller, params.numSamples)

        # For each sampled action, evaluate in environment
        # For parallel, parallelize across simulations
        if parallel is not None:
            res = parallel.starmap(envEvalFn, [(MultiEnvWrapper(env, params.numSimulationsPerSample),
                                                              timeHorizon, sampledActions[:, i],
                                                              sampledNodes[:, :, i]) for i in range(params.numSamples)])
            values, stdDev = (np.array([ent[0] for ent in res]), np.array([ent[1] for ent in res]))
        else:
            multiEnv = MultiEnvWrapper(env, params.numSimulationsPerSample)
            res = [envEvalFn(multiEnv, timeHorizon, sampledActions[:,i], sampledNodes[:,:,i]) for i in range(params.numSamples)]
            values, stdDev = (np.array([ent[0] for ent in res]), np.array([ent[1] for ent in res]))

        # Save values
        allValues[iteration, :] = values
        allStdDev[iteration, :] = stdDev

        # Find N_b best policies
        bestValues, bestSampleIndices, bestValue, bestValueVariance, controllerChange = \
            _reduceSamplesToBest(values, stdDev, bestValue, bestValueVariance, params.numBestSamples, worstValueOfPreviousIteration)

        # Update latest controller if best value changed
        if controllerChange:
            bestActionProbs = sampledActions[:, bestSampleIndices[-1]]
            bestNodeTransitionProbs = sampledNodes[:, :, bestSampleIndices[-1]]

        #If we're using a value threshold, also throw away iterations below that
        if params.valueThreshold is not None:
            bestSampleIndices = _applyValueThreshold(params.valueThreshold, bestValues, bestSampleIndices)


        # For each controller, for each node, update using best samples
        updateControllerDistribution(controller, sampledActions[:,bestSampleIndices, :], sampledNodes[:,:,bestSampleIndices, :], params.learningRate)
        print('After '+str(iteration+1) + ' iterations, best (discounted) value is ' + str(bestValue) + ' with standard deviation ' +str(bestValueVariance))
        bestValueAtEachIteration[iteration] = bestValue
        bestStdDevAtEachIteration[iteration] = bestValueVariance
        # If the value stops improving, maybe we've converged?
        if iterBestValue < bestValue+convergenceThreshold:
            iterBestValue = bestValue
        else:
            iterBestValue = bestValue
            estimatedConvergenceIteration = iteration
            # if we're using a convergence threshold, can terminate early
            if convergenceThreshold and controllerChange:
                break

        # Save occasionally so we don't lose everything in a crash. Saves relative to working dir
        if saveFrequency and iteration % saveFrequency == 0:
            saveResults(baseDir, env.spec.id, params, (bestValue, bestValueVariance, bestActionProbs, bestNodeTransitionProbs,
                                               controller, estimatedConvergenceIteration, allValues, allStdDev, bestValueAtEachIteration, bestStdDevAtEachIteration))


    # Return best policy, best value, updated controller
    return bestValue, bestValueVariance, bestActionProbs, bestNodeTransitionProbs, controller, \
           estimatedConvergenceIteration, allValues, allStdDev, bestValueAtEachIteration, bestStdDevAtEachIteration


# Return the best N_b samples. Update the best value if it changes, return whether best tables need to be updated
def _reduceSamplesToBest(sampleValues, sampleStdDev, bestValue, bestValueVariance, numBestSamples, worstValueOfPreviousIteration):
    # Find N_b best policies
    bestSampleIndices = sampleValues.argsort()[-numBestSamples:]
    bestValues = sampleValues[bestSampleIndices]
    sortedStdDev = sampleStdDev[bestSampleIndices]

    # Save best policy (if better than overall previous)
    controllerChange = False
    if bestValue < bestValues[-1]:
        controllerChange = True
        bestValue = bestValues[-1]
        bestValueVariance = sortedStdDev[-1]

    # Throw away policies below value threshold (worst best value of previous iteration)
    keepIndices = np.where(bestValues >= worstValueOfPreviousIteration)[0]
    bestValues = bestValues[keepIndices]
    bestSampleIndices = bestSampleIndices[keepIndices]
    return bestValues, bestSampleIndices, bestValue, bestValueVariance, controllerChange

# If we're using a value threshold, also throw away iterations below that
def _applyValueThreshold(valueThreshold, bestValues, bestSampleIndices):
    if valueThreshold is not None:
        keepIndices = np.where(bestValues >= valueThreshold)[0]
        bestSampleIndices = bestSampleIndices[keepIndices]
    return bestSampleIndices



