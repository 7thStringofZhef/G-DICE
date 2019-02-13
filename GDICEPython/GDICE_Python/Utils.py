import gym
import numpy as np
from .Controllers import FiniteStateControllerDistribution


# Sample actions and node transitions from controller distribution(s)
# For each node in each controller, sample actions numNodes*numSamples*numControllers(or)numAgents
# For each node, observation in controller, sample next node numObs*numBeginNodes*numSamples*numControllers(or)numAgents
def sampleFromControllerDistribution(controller, numSamples, numAgents=1):
    if isinstance(controller, (list, tuple)):
        nC = len(controller)
        return np.stack([controller[a].sampleActionFromAllNodes(numSamples) for a in range(nC)], axis=-1), \
               np.stack([controller[a].sampleAllObservationTransitionsFromAllNodes(numSamples) for a in range(nC)], axis=-1)
    else:
        # In the case of 1 controller with multiple agents, return in the same form as above
        if numAgents == 1:
            return controller.sampleActionFromAllNodes(numSamples), \
                   controller.sampleAllObservationTransitionsFromAllNodes(numSamples)
        else:
            return np.stack([controller.sampleActionFromAllNodes(numSamples) for _ in range(numAgents)], axis=-1), \
                   np.stack([controller.sampleAllObservationTransitionsFromAllNodes(numSamples) for _ in range(numAgents)], axis=-1)

# Update controller distribution(s) using sampled actions/obs and learning rate
def updateControllerDistribution(controller, sActions, sNodeObs, lr):
    if isinstance(controller, (list, tuple)):
        for a in range(len(controller)):
            controller[a].updateProbabilitiesFromSamples(sActions[:, a], sNodeObs[:, :, a], lr)
    else:
        # Note: For 1 controller with multiple agents, num samples provided should be that factor more (N_b of 5 with 2 agents becomes 10)
        controller.updateProbabilitiesFromSamples(sActions, sNodeObs, lr)

# Check the environment, return important parameters
#   Input:
#     env: Bare gym environment object
#   Outputs:
#     nAgents: Number of agents in environment
#     nActions: If nAgents is 1, number of actions. Otherwise list of number of actions
#     nObs: If nAgents is 1, number of observations. Otherwise list of number of observationss
def _checkEnv(env):
    assert isinstance(env, gym.Env)
    try:
        nAgents = env.agents
        nActions = tuple(aSpace.n for aSpace in env.action_space)
        nObs = tuple(oSpace.n for oSpace in env.observation_space)
    except:
        nAgents = 1
        nActions = env.action_space.n
        nObs = env.observation_space.n
    return nAgents, nActions, nObs

# Check the controller (or list of controllers), return important parameters
#   Input:
#     controller: FiniteStateControllerDistribution, or list or tuple of them
#   Outputs:
#     nodes: Number of nodes in controller (or list of numNodes)
#     actions: Number of actions in controller (or list of numActions)
#     obs: Number of observations in controller (or list of numObs)
def _checkControllerDist(controller):
    if isinstance(controller, (list, tuple)):
        nodes, actions, obs = zip(*[_checkSingleControllerDist(c) for c in controller])
        return nodes, actions, obs
    else:
        return _checkSingleControllerDist(controller)

def _checkSingleControllerDist(controller):
    assert isinstance(controller, FiniteStateControllerDistribution)
    return controller.numNodes, controller.numActions, controller.numObservations


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


def _parsePartialResultsToGDICERunVariables(params, results):
    bestValue, bestValueVariance, bestActionProbs, bestNodeTransitionProbs, estimatedConvergenceIteration, \
    allValues, allStdDev, bestValueAtEachIteration, bestStdDevAtEachIteration = results
    startIter = np.where(np.isnan(bestValueAtEachIteration))[0][0]  # Start after last calculated value
    worstValueOfPreviousIteration = allValues[
        startIter - 1, (np.argsort(allValues[startIter - 1, :])[-params.numBestSamples:])]
    if params.valueThreshold is None:
        worstValueOfPreviousIteration = np.min(worstValueOfPreviousIteration)
    else:
        wTemp = np.min(worstValueOfPreviousIteration)
        if wTemp < params.valueThreshold:  # If the worst value is below threshold, set to -inf
            worstValueOfPreviousIteration = np.NINF
        else:
            worstValueOfPreviousIteration = \
                np.min(worstValueOfPreviousIteration[worstValueOfPreviousIteration >= params.valueThreshold])

    return bestValue, bestValueVariance, bestActionProbs, bestNodeTransitionProbs, estimatedConvergenceIteration, \
    allValues, allStdDev, bestValueAtEachIteration, bestStdDevAtEachIteration, startIter, worstValueOfPreviousIteration






