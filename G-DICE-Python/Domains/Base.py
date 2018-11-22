from __future__ import absolute_import, print_function

import matplotlib.pyplot as plt
#from ete3 import Tree, TreeStyle, NodeStyle
import numpy as np

"""
Class to hold domain-specific dictionaries
"""
class Domain(object):
    envObsToBeliefNodeDict = {}
    envObsToEnvObsIndexDict = {}
    tmaDescriptors = []
    tmaTau = []
    tmaTerminalBeliefs = []
    tmaRewards = []
    tmaChildIndices = []
    tmaDict = {}

    """
    Constructor
    """
    def __init__(self):
        raise NotImplementedError("Must create a specific domain subclass")

    """
    Init envObs
    """
    def initXe(self):
        raise NotImplementedError("Must create a specific domain subclass")

    """
    Add a new TMA
    Input:
      tma: TMA object (contains index, descriptor, tau, terminal beliefs, rewards, allowable child tma indices
    """
    def appendToTMAs(self, tma):
        self.TMAs.append(tma)


    """
    Add a new BeliefNode
    Input:
      beliefNode: BeliefNode object (contains index, descriptor, envobs)
    """
    def appendToBeliefNodes(self, beliefNode):
        self.beliefNodes.append(beliefNode)

    """
    Add new agent to list
    Input:
      agent: Agent class to add
    """
    def appendToAgents(self, agent):
        self.agents.append(agent)

    """
    Set all agents to use a policy controller. Start them at first node
    Input:
      policyController: GraphPolicyController object
    """
    def setPolicyForAllAgents(self, policyController):
        for agent in self.agents:
            agent.setPolicy(policyController)
            agent.currentPolicyNodeIndex = 0

    """
        Evaluate the current policy
        Inputs:
          gamma: Discount factor
          numPackagesGoal: How many packages to deliver
          isOutputOn: Print debug statements
        Outputs:
          value: Value of the policy
          completionTime: Time to complete evaluation
        """

    def evaluateCurrentPolicy(self, gamma=0.99, numPackagesGoal=9999, isOutputOn=False):
        raise NotImplementedError("Must create a specific domain subclass")

    """
    Print allowable TMAs from all TMA indices
    """

    def printAllowableTMAs(self):
        for TMAIndex in range(1, self.numTMAs + 1):
            print(TMAIndex)
            print(self.TMAs[TMAIndex - 1].allowableChildTMAIndices)

    """
    Construct a high-level policy tree from TMAs
    """

    def constructTree(self):
        for TMA in self.TMAs:
            if np.isnan(TMA.bTerm):  # NaN means term set = init set
                # Find allowable init set elements
                allTerminationIndices = np.where(not np.isnan(TMA.tau)).flatten()
                for index in allTerminationIndices:
                    TMA.allowableChildTMAIndices = self.findAllowableTMAs(index)
            else:  # Otherwise, specific termination set for this TMA
                for beliefIndex in range(self.numBeliefNodes):
                    if TMA.bTerm[beliefIndex] == 1:
                        TMA.allowableChildTMAIndices = self.findAllowableTMAs(beliefIndex)

    """
    Return a list of the allowable TMAs given a belief node
    Input:
      initiationIndex: Index of belief node. 0-numNodes-1
    Output:
      allowableTMAs: List of allowable TMAs
    """

    def findAllowableTMAs(self, initiationIndex):
        return [tma for tma in self.TMAs if not np.isnan(tma.tau[initiationIndex])]

    """
    """

    def drawTree(self, rootNodes, rootNodeTMAIndices, nodeMap, nodeMapIndices, maxDepth, numLeaves):
        labelTMAs = False

        numLeaves, nodeMap, nodeMapIndices = self.createDrawnTree(rootNodes, rootNodeTMAIndices, nodeMap,
                                                                  nodeMapIndices, maxDepth, numLeaves)
        """
        t = Tree()
        tStyle = TreeStyle()
        nStyle = NodeStyle()
        nStyle["shape"] = "sphere"
        nStyle["size"] = 10
        nStyle["fgcolor"] = "darkred"
        tStyle.rotation = 90 #Rotate 90 degrees
        treeFigure = plt.Figure()

        if labelTMAs:
            pass
        """

    """
    Create a TMA policy tree to be drawn in "drawTree"
    Inputs:
      Same as above
    Outputs:
      numLeaves: Number of leaves in created tree
      nodeMap: Updated nodeMap
      nodeMapIndices: Updated nodeMapIndices
    """
    def createDrawnTree(self, rootNodes, rootNodeTMAIndices, nodeMap, nodeMapIndices, maxDepth, numLeaves):
        if numLeaves < 0:
            numLeaves = 0

        if maxDepth <= 0:
            return numLeaves, nodeMap, nodeMapIndices

        numNodes = nodeMap.shape[0]
        if numNodes == 0:
            nodeMap = [0]
            nodeMapIndices = rootNodes

        tempIndex = 0
        for rootNode in rootNodes:
            rootNodeTMAIndex = rootNodeTMAIndices[tempIndex]
            # Get all children of the root node
            rootNodeTMAIndicesNext = self.TMAs[rootNodeTMAIndex - 1].allowableChildTMAIndices
            numChildren = rootNodeTMAIndicesNext.shape[0]
            rootNodesNext = np.arange(numChildren) + nodeMap.shape[0]  # Tree index for visualization
            nodeMap = nodeMap.append(np.ones(numChildren) * rootNode)
            nodeMapIndices = nodeMapIndices.append(rootNodeTMAIndicesNext)

            # Go to next level
            numLeaves, nodeMap, nodeMapIndices = self.createDrawnTree(rootNodesNext, rootNodeTMAIndicesNext, nodeMap,
                                                                      nodeMapIndices, maxDepth - 1, numLeaves)
            tempIndex += 1

        # If maxDepth is 1, we are at leaf level. Report # leaves
        if maxDepth == 1:
            tempIndex = 0
            for rootNode in rootNodes:
                rootNodeTMAIndex = rootNodeTMAIndices[tempIndex]
                # Get all children of root node
                rootNodesTMAIndicesNext = self.TMAs[rootNodeTMAIndex - 1].allowableChildTMAIndices
                numLeaves += rootNodeTMAIndicesNext.shape[0]
                tempIndex += 1


"""
Class to represent environmental observations in various domains

"""
class EnvObs(object):

    """
    Constructor
    """
    def __init__(self):
        raise NotImplementedError("Must create subclass of EnvObs")