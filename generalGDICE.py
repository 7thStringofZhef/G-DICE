import os
import gym
import numpy as np
from multiprocessing import Pool
from functools import partial
from GDICE_Python.Domains import MultiActionPOMDP
from GDICE_Python.Controllers import FiniteStateControllerDistribution, DeterministicFiniteStateController
import pickle


# Run GDICE with controller(s) on an environment, given
# Inputs:
#   env: Gym-like environment to evaluate on
#   controller: A controller or list of controllers corresponding to agents in the environment
#   params: GDICEParams object
#   timeHorizon: Number of timesteps to evaluate to. If None, run each sample until episode is finished
#   parallel: Attempt to use python multiprocessing across samples. If not None, should be a Pool object

def runGDICEOnEnvironment(env, controller, params, timeHorizon=50, parallel=None, convergenceThreshold=0):
    # Ensure controller matches environment
    assert env.action_space.n == controller.numActions
    assert env.observation_space.n == controller.numObservations

    # Reset controller
    controller.reset()

    # Start variables
    bestActionProbs = None
    bestNodeTransitionProbs = None
    bestValue = np.NINF
    bestValueVariance = 0
    worstValueOfPreviousIteration = np.NINF
    allValues = np.zeros((params.numIterations, params.numSamples), dtype=np.float64)
    allStdDev = np.zeros((params.numIterations, params.numSamples), dtype=np.float64)
    estimatedConvergenceIteration = 0

    for iteration in range(params.numIterations):
        controllerChange = False  # Did the controller change this iteration?
        iterBestValue = np.NINF  # What is the most recently seen best controller value
        # For each node in controller, sample actions
        sampledActions = controller.sampleActionFromAllNodes(params.numSamples)  # numNodes*numSamples

        # For each node, observation in controller, sample next node
        sampledNodes = controller.sampleAllObservationTransitionsFromAllNodes(params.numSamples)  # numObs*numBeginNodes*numSamples

        # For each sampled action, evaluate in environment
        # For parallel, try single environment. For single core (or low memory), use MultiEnv
        if parallel is not None and isinstance(parallel, type(Pool)):
            envEvalFn = partial(evaluateSample, timeHorizon=timeHorizon, numSimulations=params.numSimulationsPerSample)
            values, stdDev = [(np.array(res[0]), np.array(res[1])) for res in parallel.starmap(envEvalFn, [(env, sampledActions[:,i], sampledNodes[:,:,i]) for i in range(params.numSamples)])]
        else:
            values, stdDev = evaluateSamplesMultiEnv(MultiActionPOMDP(env,params.numSamples), timeHorizon, params.numSimulationsPerSample, sampledActions, sampledNodes)

        # Save values
        allValues[iteration, :] = values
        allStdDev[iteration, :] = stdDev

        # Find N_b best policies
        bestSampleIndices = values.argsort()[-params.numBestSamples:]
        bestValues = values[bestSampleIndices]
        sortedStdDev = stdDev[bestSampleIndices]

        # Save best policy (if better than overall previous)
        if bestValue < bestValues[-1]:
            controllerChange = True
            bestValue = bestValues[-1]
            bestValueVariance = sortedStdDev[-1]
            bestActionProbs = sampledActions[:, bestSampleIndices[-1]]
            bestNodeTransitionProbs = sampledNodes[:, :, bestSampleIndices[-1]]

        # Throw away policies below value threshold (worst best value of previous iteration)
        keepIndices = np.where(bestValues >= worstValueOfPreviousIteration)[0]
        bestValues = bestValues[keepIndices]
        bestSampleIndices = bestSampleIndices[keepIndices]

        #If we're using a value threshold, also throw away iterations below that
        if params.valueThreshold is not None:
            keepIndices = np.where(bestValues >= params.valueThreshold)[0]
            bestValues = bestValues[keepIndices]
            bestSampleIndices = bestSampleIndices[keepIndices]


        # For each node, update using best samples
        controller.updateProbabilitiesFromSamples(sampledActions[:,bestSampleIndices], sampledNodes[:,:,bestSampleIndices], params.learningRate)
        print('After '+str(iteration+1) + ' iterations, best (discounted) value is ' + str(bestValue) + 'with standard deviation '+str(bestValueVariance))

        # If the value stops improving, maybe we've converged?
        if iterBestValue < bestValue+convergenceThreshold:
            iterBestValue = bestValue
        else:
            iterBestValue = bestValue
            estimatedConvergenceIteration = iteration
            # if we're using a convergence threshold, can terminate early
            if convergenceThreshold and controllerChange:
                break

    # Return best policy, best value, updated controller
    return bestValue, bestValueVariance, bestActionProbs, bestNodeTransitionProbs, controller, \
           estimatedConvergenceIteration, allValues, allStdDev


# Evaluate a single sample, starting from first node
# Inputs:
#   env: Environment in which to evaluate
#   timeHorizon: Time horizon over which to evaluate
#   actionTransitions: (numNodes,) int array of chosen actions for each node
#   nodeObservationTransitions: (numObs, numNodes) int array of chosen node transitions for obs
#  Output:
#    value: Discounted total return over timeHorizon (or until episode is done), averaged over all simulations
#    stdDev: Standard deviation of discounter total returns over all simulations
def evaluateSample(env, timeHorizon, numSimulations, actionTransitions, nodeObservationTransitions):
    gamma = env.discount
    values = np.zeros(numSimulations, dtype=np.float64)
    for sim in numSimulations:
        env.reset()
        currentNodeIndex = 0
        currentTimestep = 0
        isDone = False
        value = 0.0
        while not isDone and currentTimestep < timeHorizon:
            obs, reward, isDone = env.step(actionTransitions[currentNodeIndex])
            currentNodeIndex = nodeObservationTransitions[obs, currentNodeIndex]
            value += reward * (gamma ** currentTimestep)
            currentTimestep += 1
        values[sim] = value
    return values.mean(), values.std()

# Evaluate multiple samples, starting from first node
# Inputs:
#   env: MultiEnv environment in which to evaluate
#   timeHorizon: Time horizon over which to evaluate
#   actionTransitions: (numNodes,numSamples) int array of chosen actions for each node
#   nodeObservationTransitions: (numObs, numNodes, numSamples) int array of chosen node transitions for obs
#  Output:
#    allSampleValues: Discounted total return over timeHorizon (or until episode is done), averaged over all simulations, for each sample (numSamples,)
#    stdDevs: Standard deviation of discounted total returns over all simulations, for each sample (numSamples,)
def evaluateSamplesMultiEnv(env, timeHorizon, numSimulations, actionTransitions, nodeObservationTransitions):
    assert isinstance(env, MultiActionPOMDP)
    gamma = env.discount
    numSamples = actionTransitions.shape[-1]
    sampleIndices = np.arange(numSamples)
    allSampleValues = np.zeros((numSimulations, numSamples), dtype=np.float64)
    for sim in range(numSimulations):
        env.reset()
        currentNodes = np.zeros(numSamples, dtype=np.int32)
        currentTimestep = 0
        values = np.zeros(numSamples, dtype=np.float64)
        isDones = np.zeros(numSamples, dtype=bool)
        while not all(isDones) and currentTimestep < timeHorizon:
            obs, rewards, isDones = env.step(actionTransitions[currentNodes, sampleIndices])[:3]
            currentNodes = nodeObservationTransitions[obs, currentNodes, sampleIndices]
            values += rewards * (gamma ** currentTimestep)
            currentTimestep += 1
        allSampleValues[sim, :] = values
    return allSampleValues.mean(axis=0), allSampleValues.std(axis=0)



# GDICE parameter object
# Inputs:
#   numIterations: N_k number of iterations of GDICE to perform
#   numSamples: N_s number of samples to take for each iteration from each node
#   numSimulationsPerSample: Number of times to run the environment for each sampled controller. Values will be averaged over these runs
#   numBestSamples: N_b number of samples to keep from each set of samples
#   learningRate: 0-1 alpha value, learning rate at which controller shifts probabilities
#   valueThreshold: If not None, ignore all samples with worse values, even if that means there aren't numBestSamples
class GDICEParams(object):
    def __init__(self, numIterations=30, numSamples=50, numSimulationsPerSample=1000, numBestSamples=5, learningRate=0.1, valueThreshold=None):
        self.numIterations = numIterations
        self.numSamples = numSamples
        self.numSimulationsPerSample = numSimulationsPerSample
        self.numBestSamples = numBestSamples
        self.learningRate = learningRate
        self.valueThreshold = valueThreshold
        self.buildName()

    # Name for use in saving files
    def buildName(self):
        self.name = 'K' + str(self.numIterations) + '_S' + str(self.numSamples) + '_sim' + \
                    str(self.numSimulationsPerSample) + '_B' + str(self.numBestSamples) + '_lr' + \
                    str(self.learningRate) + '_vT' + 'None' if self.valueThreshold is None else str(self.valueThreshold)


if __name__ == "__main__":
    envName = 'POMDP-4x3-episodic-v0'
    env = gym.make(envName)  # Make a gym environment with POMDP-1d-episodic-v0
    controllerDistribution = FiniteStateControllerDistribution(10, env.action_space.n, env.observation_space.n)  # make a controller with 10 nodes, with #actions and observations from environment
    testParams = GDICEParams()  # Choose G-DICE parameters (look above for explanation)
    #pool = Pool()  # Use a pool for parallel processing. Max # threads
    pool = None  # use a multiEnv for vectorized processing on computers with low memory or no core access

    # Run GDICE. Return the best average value, its standard deviation,
    # tables of the best deterministic transitions, and the updated distribution of controllers
    bestValue, bestValueStdDev, bestActionTransitions, bestNodeObservationTransitions, updatedControllerDistribution, \
    estimatedConvergenceIteration, allValues, allStdDev = \
        runGDICEOnEnvironment(env, controllerDistribution, testParams, timeHorizon=50, parallel=pool)

    # Save
    """
    os.mkdir('GDICEresults')
    savePath = os.path.join('GDICEResults', envName)  # relative to current path
    os.mkdir(savePath)
    np.savez(os.path.join(savePath, testParams.name)+'.npz', bestValue=bestValue, bestValueStdDev=bestValueStdDev,
             bestActionTransitions=bestActionTransitions, bestNodeObservationTransitions=bestNodeObservationTransitions,
             estimatedConvergenceIteration=estimatedConvergenceIteration, allValues=allValues, allStdDev=allStdDev)
    pickle.dump(updatedControllerDistribution, os.path.join(savePath, testParams.name)+'.pkl', 'wb')
    """

    # Create a deterministic controller from the tables above
    bestDeterministicController = DeterministicFiniteStateController(bestActionTransitions, bestNodeObservationTransitions)

    # Test on environment

