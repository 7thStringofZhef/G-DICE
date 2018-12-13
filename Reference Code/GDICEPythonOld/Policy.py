from __future__ import absolute_import, print_function, division

import numpy as np
"""
Class to represent an Agent's policy

Instance variables
  table: Table containing policy transitions
"""
class Policy(object):

    """
    Constructor
    """
    def __init__(self, table):
        self.table = table

    """
    Get the next TMA Index
    """
    def getNextTMAIndex(self, currentTMAIndex, currentXeIndex):
        raise NotImplementedError('Must be called from subclass')

"""
Class to represent a node in a GraphPolicyController

Instance variables:
  nodeIndex: Index of node
  nodeTMA: TMA to be executed in this node
  nextNode: Array of indices to nodes that may follow this one. 1*numObs
  pVectorTMA: Array of probabilities for picking next TMA. NumTMAs*1
  pTableNextNode: numNodesInFullGraph*numObs matrix, probability of choosing next graph node based on envObs object
  numTMAs
  numObs
  numNodesInFullGraph
  TMAs
  transitions: Matrix of sampled next-node transitions
"""
class GraphNode(object):

    """
    Constructor
    """
    def __init__(self, numNodesInFullGraph, nodeIndex, numTMAs, numObs, numSamples):
        self.nodeIndex = nodeIndex
        self.numNodesInFullGraph = numNodesInFullGraph
        self.numObs = numObs
        self.numTMAs = numTMAs
        self.transitions = np.zeros((numSamples, numObs), dtype=np.int32)
        self.pVectorTMA = np.ones((numTMAs,1)) / numTMAs
        self.pTableNextNode = np.ones((numNodesInFullGraph, numObs)) / numNodesInFullGraph
        self.nextNode = None

    """
    Sample TMAs from TMA distribution
    Input:
      numSamples: Number of samples to take
    """
    def sampleTMAs(self, numSamples):
        self.TMAs = np.random.choice(range(1, self.numTMAs+1), size=numSamples, p=self.pVectorTMA.flatten())

    """
    Sample transitions from transition table for a particular environmental observation
    Input: 
      observationIndex: Index of environmental observation in domain
      numSamples: Number of samples
    """
    def sampleTransitions(self, observationIndex, numSamples):
        self.transitions[:, observationIndex-1] = np.random.choice(range(self.numNodesInFullGraph), size=numSamples,
                                                                   p=self.pTableNextNode[:, observationIndex-1].flatten())
    """
    Set TMA and next node to this sample index
    Input:
      sampleIndex: Index of TMA
    """
    def setToSampleNumber(self, sampleIndex):
        self.nodeTMA = self.TMAs[sampleIndex]
        self.nextNode = self.transitions[sampleIndex, :]

    """
    Set TMA
    Input:
      TMA: TMA index to set to
    """
    def setTMA(self, TMA):
        self.nodeTMA = TMA


    """
    Set the next node to goto, given this environmental observation
    Input:
      observationIndex: Index of environmental observation
      nextNodeIndex: Index of next node
    """
    def setNextNode(self, observationIndex, nextNodeIndex):
        self.nextNode[observationIndex-1] = nextNodeIndex




"""
Class to represent a FSA policy
Instance variables:
  nodes: Nodes in the FSC. Indexed 0-numNodes-1
  numNodes: Number of nodes in the FSC
  alpha: learning rate
  numObs: Number of observations
  numTMAs: Number of TMAs in the graph
  numSamples: Number of samples
  
"""
class GraphPolicyController(object):


    """
    Constructor
    """
    def __init__(self, numNodes, alpha, numTMAs, numObs, numSamples):
        self.numNodes = numNodes
        self.alpha = alpha
        self.numObs = numObs
        self.numTMAs = numTMAs
        self.numSamples = numSamples
        self.nodes = []  # Empty list of nodes
        for nodeIndex in range(numNodes):
            self.appendNodeToGraph(nodeIndex, numTMAs, numObs, numSamples)

    """
    Sample TMAs at the node, using updated probability density function of best nodes
    Then sample transitions.
    Inputs:
      numSamples: How many samples to take
    """
    def sample(self, numSamples):
        for node in self.nodes:
            node.sampleTMAs(numSamples)
            for observationIndex in range(1, self.numObs+1):
                node.sampleTransitions(observationIndex, numSamples)

    """
    Set the TMA to be executed at the node, and next-node transition
    Input:
      sampleIndex: Index of sample (TMA, nextNode) to set all nodes to
    """
    def setGraph(self, sampleIndex):
        for node in self.nodes:
            node.setToSampleNumber(sampleIndex)

    """
    Update policy probabilities
    Inputs:
      currentIterationValues: Values of samples
      N_b: Number n best samples to keep
      isOutputOn: Print debug messages
    """
    def updateProbs(self, currentIterationValues, N_b, isOutputOn=True):
        # Sort in descending order, get N_b best values > 0
        sortedValues, sortingIndices = np.sort(currentIterationValues)[::-1], currentIterationValues[::-1].argsort()
        maxValues = sortedValues[:N_b]
        maxValueIndices = sortingIndices[:N_b]
        maxValueIndices = maxValueIndices[maxValues > 0]  # Remove below 0
        N_b = maxValueIndices.shape[0]
        if isOutputOn:
            print(N_b,' samples with value > 0 found!')

        #Ensure we have at least 1 sample so we don't need to renormalize pdf
        if N_b == 0:
            return

        weightPerSample = 1 / N_b
        # ***Could make this more efficient if I can vectorize***
        for node in self.nodes:
            newPVectorTMA = node.pVectorTMA * (1-self.alpha)
            newPTableNextNode = node.pTableNextNode * (1-self.alpha)
            for sampleIndex in maxValueIndices:
                if isOutputOn:
                    print('Updating weights using "best" sample ', sampleIndex)

                sampleTMA = node.TMAs[sampleIndex]
                newPVectorTMA[sampleTMA-1] = newPVectorTMA[sampleTMA-1] + weightPerSample*self.alpha

                for observationIndex in range(self.numObs):
                    sampleNextNode = node.transitions[sampleIndex, observationIndex]
                    newPTableNextNode[sampleNextNode, observationIndex] = newPTableNextNode[sampleNextNode, observationIndex] + weightPerSample*self.alpha

            #Update the pdfs
            node.pVectorTMA = newPVectorTMA
            node.pTableNextNode = newPTableNextNode



    """
    Add a new node to the graph
    Inputs:
      nodeIndex: Index of node
      numTMAs: Number of task macro-actions
      numObs: number of observations
      numSamples: Number of samples
    """
    def appendNodeToGraph(self, nodeIndex, numTMAs, numObs, numSamples):
        self.nodes.append(GraphNode(self.numNodes, nodeIndex, numTMAs, numObs, numSamples))


    """
    Print this policy controller
    """
    def __str__(self):
        print('GraphPolicyController nodes: ' ' '.join([node.nodeTMA for node in self.nodes]))

    """
    Get the next TMA index given the current node and observation
    Inputs:
      currentNodeIndex: Index of current node in controller
      currentObservationIndex: Index of environmental observation in domain
    Outputs:
      newPolicyNodeIndex: Index of next policy node
      newTMAIndex: Index of next TMA
    """
    def getNextTMAIndex(self, currentNodeIndex, currentObservationIndex):
        newPolicyNodeIndex = self.nodes[currentNodeIndex].nextNode[currentObservationIndex-1]
        newTMAIndex = self.nodes[newPolicyNodeIndex].nodeTMA
        return newPolicyNodeIndex, newTMAIndex

    """
    Return the policy table for this controller
    Outputs:
      TMAs: TMAs represented in this policy
      transitions: Node transitions in this policy
    """
    def getPolicyTable(self):
        TMAs = np.zeros(self.numNodes)
        transitions = np.zeros((self.numNodes, self.numObs), dtype=np.int32)
        for node in self.nodes:
            TMAs[node.nodeIndex] = node.nodeTMA
            transitions[node.nodeIndex, :] = node.nextNode
        return TMAs, transitions

    """
    Set the policy according to variables as retrieved from above
    Inputs:
      TMAs: TMAs represented in this policy
      transitions: Node transitions in this policy
    """
    def setPolicyUsingTables(self, TMAs, transitions):
        for node in self.nodes:
            node.nodeTMA = TMAs[node.nodeIndex]
            node.nextNode = transitions[node.nodeIndex, :]



