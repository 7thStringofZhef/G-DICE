import numpy as np
import numpy.random as npr
from scipy.stats import entropy


# Get columnwise entropy for a probability table (rows*cols)
def getColumnwiseEntropy(pTable, nCols):
    return np.array([entropy(pTable[:, col]) for col in range(nCols)])


# Get maximum entropy value for a number of rows
def getMaximalEntropy(nRows):
    return entropy(np.ones(nRows)/nRows)


# Class to sample finite state controllers from
# Provides an interface to sample possible action and nodeObservation transitions
# Inputs:
#   numNodes: Number of nodes in the controllers
#   numActions: Number of actions controller nodes can perform (should match environment)
#   numObservations: Number of observations a controller can see (should match environment)
class FiniteStateControllerDistribution(object):
    def __init__(self, numNodes, numActions, numObservations, shouldInjectNoiseUsingMaximalEntropy=False, noiseInjectionRate=0.05, entFraction=0.02):
        self.numNodes = numNodes
        self.numActions = numActions
        self.numObservations = numObservations
        self.currentNode = None
        self.shouldInjectNoiseUsingMaximalEntropy = shouldInjectNoiseUsingMaximalEntropy
        self.entFraction = entFraction
        self.noiseInjectionRate = noiseInjectionRate
        self.initActionNodeProbabilityTable()
        self.initObservationNodeTransitionProbabilityTable()

    # Probability of each action given being in a certain node
    def initActionNodeProbabilityTable(self):
        initialProbability = 1 / self.numActions
        self.actionProbabilities = np.full((self.numNodes, self.numActions), initialProbability)

    # Probability of transition from 1 node to second node given observation
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
        if actions.size == 0:  # No samples, no update
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
        obsIndices = np.arange(0, self.numObservations, dtype=int)

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

    # Inject noise into probability table (entropy injection)
    # actionProbabilities is (numNodes, numActions)
    # obsProbabilities is (numNodes, numNodes, numObservations)
    """
    # One node pTableTMA is numActions*numObs
    # One node pTableNextNode is numNodes*numObs
    for idxNode = 1:length(obj.nodes)  # (Start node)
        entColumnsPTableTMA = entropy_columnwise(obj.nodes(idxNode).pTableTMA,obj.numObs);  # Entropy across action probabilities for each observation
        idxsDegenPTableTMA = entColumnsPTableTMA < maxEnt*entFractionForInjection;  # Observation indices at which entropy is too low
        
        # For each of those columns, inject entropy
        obj.nodes(idxNode).pTableTMA(:,idxsDegenPTableTMA) = (1-noise_injection_rate)*obj.nodes(idxNode).pTableTMA(:,idxsDegenPTableTMA) + noise_injection_rate*ones(obj.numTMAs,sum(idxsDegenPTableTMA))./obj.numTMAs;
        
        entColumnsPTableNextNode = entropy_columnwise(obj.nodes(idxNode).pTableNextNode,obj.numObs);  # Entropy across node probabilities for each observation
        idxsDegenPTableNextNode = entColumnsPTableNextNode < maxEnt*entFractionForInjection;  # Observation indices at which entropy is too low
        
        # For each of those columns, inject entropy
        obj.nodes(idxNode).pTableNextNode(:,idxsDegenPTableNextNode) = (1-noise_injection_rate)*obj.nodes(idxNode).pTableNextNode(:,idxsDegenPTableNextNode) + noise_injection_rate*ones(obj.numNodes,sum(idxsDegenPTableNextNode))./obj.numNodes;
        
        if (sum(idxsDegenPTableTMA)>1 || sum(idxsDegenPTableNextNode)>1)
%                         fprintf(['idxNode: ' num2str(idxNode) '| sum(idxsDegenPTableTMA): ' num2str(sum(idxsDegenPTableTMA)) ' | sum(idxsDegenPTableNextNode): ' num2str(sum(idxsDegenPTableNextNode)) '\n'])
            just_injected_noise = true;
        end
    end
    """
    def injectNoise(self):
        injectedNoise = False
        nodeIndices = np.arange(self.numNodes)
        obsIndices = np.arange(self.numObservations)
        actionIndices = np.arange(self.numActions)
        if self.shouldInjectNoiseUsingMaximalEntropy:
            maxEntropy = getMaximalEntropy(self.numNodes)  # Maximum entropy for categorical pdf
            noiseInjectionRate = self.noiseInjectionRate  # Rate (0 to 1) at which to inject noise
            entropyFractionForInjection = self.entFraction  # Threshold of max entropy required to inject

            # Inject entropy into action probabilities. Does this make sense for moore machines?
            # Makes sense for moore. Imagine that action tables have one observation. You just need to say whether entropy of actions for each node is sufficient
            actionEntropy = np.array([entropy(self.actionProbabilities[idx, :], base=2) for idx in nodeIndices])
            nIndices = actionEntropy < maxEntropy * entropyFractionForInjection  # numNodes,
            self.actionProbabilities[nIndices, :] = (1-noiseInjectionRate)*self.actionProbabilities[nIndices, :] + \
                                                    noiseInjectionRate*np.ones(np.sum(nIndices), self.numActions)/self.numActions


            # Inject entropy into node transition probabilities
            for startNodeIdx in nodeIndices:
                nodeEntropyPerObs = getColumnwiseEntropy(self.nodeTransitionProbabilities[startNodeIdx, :, :], self.numObservations)
                ntIndices = nodeEntropyPerObs < maxEntropy * entropyFractionForInjection  # numObs,
                # Subsection will be 10 * numColsToInject
                self.nodeTransitionProbabilities[startNodeIdx, :, ntIndices] \
                    = (1-noiseInjectionRate) * \
                      self.nodeTransitionProbabilities[startNodeIdx, :, ntIndices] + \
                      noiseInjectionRate * np.ones((np.sum(ntIndices), self.numNodes))/self.numNodes
        injectedNoise = True
        return injectedNoise





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
        return self.actionTransitions[self.currentNode]

    # Set current node using observation
    def processObservation(self, observationIndex):
        self.currentNode = self.nodeObservationTransitions[observationIndex, self.currentNode]

    # return current node index
    def getCurrentNode(self):
        return self.currentNode
