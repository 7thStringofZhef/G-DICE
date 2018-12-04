from gym_pomdps import list_pomdps
import gym
import numpy as np
import numpy.random as npr

class FiniteStateController(object):
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

    # Get the next node according to probability given current node and observation index
    # Can sample multiple transitions
    # DOES NOT set the current node
    def sampleObservationTransition(self, observationIndex, numSamples=1):
        return npr.choice(np.arange(self.numActions), size=numSamples, p=self.actionProbabilities[self.currentNode, :, observationIndex])


    # Update the probability of taking an action in a particular node
    # Can be used for multiple inputs if numNodeIndices = n, numActionIndices = m, and newProbability = n*m or a scalar
    def updateActionProbability(self, nodeIndex, actionIndex, newProbability):
        self.actionProbabilities[nodeIndex, actionIndex] = newProbability

    # Update the probability of transitioning from one node to a second given an observation
    def updateTransitionProbability(self, firstNodeIndex, secondNodeIndex, observationIndex, newProbability):
        self.nodeTransitionProbabilities[firstNodeIndex, secondNodeIndex, observationIndex] = newProbability






if __name__ == "__main__":
    env = gym.make(list_pomdps()[2])  # 4x3-v0 POMDP