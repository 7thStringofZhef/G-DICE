import numpy as np
import numpy.random as npr


# Class to sample finite state controllers from
# Provides an interface to sample possible action and nodeObservation transitions
# Inputs:
#   numNodes: Number of nodes in the controllers
#   numActions: Number of actions controller nodes can perform (should match environment)
#   numObservations: Number of observations a controller can see (should match environment)
class FiniteStateControllerDistribution(object):
    def __init__(self, numNodes, numActions, numObservations):
        self.numNodes = numNodes
        self.numActions = numActions
        self.numObservations = numObservations
        self.currentNode = None
        self.initActionNodeProbabilityTable()
        self.initObservationNodeTransitionProbabilityTable()

    # Probability of each action given being in a certain node
    def initActionNodeProbabilityTable(self):
        initialProbability = 1 / self.numActions
        self.actionProbabilities = np.full((self.numNodes, self.numActions), initialProbability)

    # Probability of transition from 1 node to second node given obsersvation
    def initObservationNodeTransitionProbabilityTable(self):
        initialProbability = 1 / self.numNodes
        self.nodeTransitionProbabilities = np.full((self.numNodes,
                                                    self.numNodes,
                                                    self.numObservations), initialProbability)

    # Set the current node of the controller
    def setNode(self, nodeIndex):
        self.currentNode = nodeIndex

    # Returns the index of the current node
    def getCurrentNode(self):
        return self.currentNode

    # Reset the controller to default probabilities
    def reset(self):
        self.setNode(None)
        self.initActionNodeProbabilityTable()
        self.initObservationNodeTransitionProbabilityTable()

    # Get an action using the current node according to probability. Can sample multiple actions
    def sampleAction(self, numSamples=1):
        return npr.choice(np.arange(self.numActions), size=numSamples, p=self.actionProbabilities[self.currentNode, :])

    # Get an action from all nodes according to probability. Can sample multiple actions
    # Outputs numNodes * numSamples
    def sampleActionFromAllNodes(self, numSamples=1):
        actionIndices = np.arange(self.numActions)
        return np.array([npr.choice(actionIndices, size=numSamples, p=self.actionProbabilities[nodeIndex,:])
                         for nodeIndex in range(self.numNodes)], dtype=np.int32)

    # Get the next node according to probability given current node and observation index
    # Can sample multiple transitions
    # DOES NOT set the current node
    def sampleObservationTransition(self, observationIndex, numSamples=1):
        return npr.choice(np.arange(self.numNodes), size=numSamples, p=self.nodeTransitionProbabilities[self.currentNode, :, observationIndex])

    # Get the next node for each node given observation index
    # Outputs numNodes * numSamples
    def sampleObservationTransitionFromAllNodes(self, observationIndex, numSamples=1):
        nodeIndices = np.arange(self.numNodes)
        return np.array([npr.choice(nodeIndices, size=numSamples, p=self.nodeTransitionProbabilities[nodeIndex, :, observationIndex])
                         for nodeIndex in range(self.numNodes)], dtype=np.int32)

    # Get the next node for all nodes for all observation indices
    # Outputs numObs * numNodes * numSamples
    def sampleAllObservationTransitionsFromAllNodes(self, numSamples=1):
        obsIndices = np.arange(self.numObservations)
        return np.array([self.sampleObservationTransitionFromAllNodes(obsIndex, numSamples)
                         for obsIndex in obsIndices], dtype=np.int32)

    def updateProbabilitiesFromSamples(self, actions, nodeObs, learningRate):
        if len(actions) == 0:  # No samples, no update
            return
        assert actions.shape[-1] == nodeObs.shape[-1]  # Same # samples
        if len(actions.shape) == 1:  # 1 sample
            weightPerSample = 1
            numSamples = 1
            actions = np.expand_dims(actions, axis=1)
            nodeObs = np.expand_dims(nodeObs, axis=2)
        else:
            weightPerSample = 1/actions.shape[-1]
            numSamples = actions.shape[-1]

        # Reduce
        self.actionProbabilities = self.actionProbabilities * (1-learningRate)
        self.nodeTransitionProbabilities = self.nodeTransitionProbabilities * (1-learningRate)
        nodeIndices = np.arange(0, self.numNodes, dtype=int)
        obsIndices = np.arange(0,self.numObservations, dtype=int)

        # Add samples factored by weight
        for sample in range(numSamples):
            self.actionProbabilities[nodeIndices, actions[:,sample]] += learningRate*weightPerSample
            #self.nodeTransitionProbabilities[nodeIndices, nodeObs[repObsIndices, nodeIndices, sample], obsIndices] += learningRate*weightPerSample
            for observation in range(nodeObs.shape[0]):
                for startNode in range(nodeObs.shape[1]):
                    self.nodeTransitionProbabilities[startNode, nodeObs[observation,startNode,sample], observation] += learningRate*weightPerSample

    # Update the probability of taking an action in a particular node
    # Can be used for multiple inputs if numNodeIndices = n, numActionIndices = m, and newProbability = n*m or a scalar
    def updateActionProbability(self, nodeIndex, actionIndex, newProbability):
        self.actionProbabilities[nodeIndex, actionIndex] = newProbability

    # Update the probability of transitioning from one node to a second given an observation
    def updateTransitionProbability(self, firstNodeIndex, secondNodeIndex, observationIndex, newProbability):
        self.nodeTransitionProbabilities[firstNodeIndex, secondNodeIndex, observationIndex] = newProbability

    # Get the probability vector for node(s)
    def getPolicy(self, nodeIndex):
        return self.actionProbabilities[np.array(nodeIndex, dtype=np.int32), :]

    # Get the current probability tables
    def save(self):
        return self.actionProbabilities, self.nodeTransitionProbabilities


# A deterministic FSC that has one action for any node and one end node transition for any node-obs combination
# Constructed using output policy from G-DICE
#   Inputs:
#     actionTransitions: (numNodes, ) array of actions to perform at each node
#     nodeObservationTransitions: (numObservations, numNodes) array of end nodes to transition to
#                                 from each start node and observation combination
class DeterministicFiniteStateController(object):
    def __init__(self, actionTransitions, nodeObservationTransitions):
        self.actionTransitions = actionTransitions
        self.nodeObservationTransitions = nodeObservationTransitions
        self.numNodes = self.actionTransitions.shape[0]
        self.numActions = np.unique(self.actionTransitions)
        self.numObservations = self.nodeObservationTransitions[0]
        self.currentNode = 0

    # Set current node to 0
    def reset(self):
        self.currentNode = 0

    # Get action using current node
    def getAction(self):
        return self.actionTransitions[self.currentNodes]

    # Set current node using observation
    def processObservation(self, observationIndex):
        self.currentNode = self.nodeObservationTransitions[observationIndex, self.currentNodes]

    # return current node index
    def getCurrentNode(self):
        return self.currentNode
