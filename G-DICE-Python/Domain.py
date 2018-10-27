from __future__ import absolute_import, print_function, division

import numpy as np

from .TMA import TMA
from .BeliefNode import BeliefNode
from .Policy import GraphPolicyController

class Domain(object):

    def __init__(self):
        self.beliefNodes = [] #List of belief nodes in this domain
        self.TMAs = [] #List of task macro-actions in this domain
        self.numTMAs = 0 #Initially no TMAs
        self.numBeliefNodes = 5 #Number of expected belief nodes
        self.agents = [] #List of agents in domain
        self.validTMATransitions


    """
    Set agents in xE, init packages at bases
    """
    def initXe(self):
        for agent in self.agents:
            agent.currentBeliefNode.xeSetPhi(agent.agentIndex, 1, 0)
        for baseIndex in [1, 2]:
            self.beliefNodes[baseIndex-1].xeSamplePackage()

    """
    Add a new TMA
    Inputs:
      tmaIndex: Index of TMA, 1-numTMAs+1
      tmaDescriptor: String describing the tma
      tau: List of expected completion times of TMA (for each belief node)
      bTerm: List of belief nodes, 0 if not a terminal node, 1 if terminal, NaN if TMA terminates in same node it started in
      rewards: List of rewards (for each belief node)
      allowableChildTMAs: list of possible subsequent TMAs
    """
    def appendToTMAs(self, tmaIndex, tmaDescriptor, tau, bTerm, rewards, allowableChildTMAIndices):
        self.TMAs.append(TMA(tmaIndex, tmaDescriptor, tau, bTerm, rewards, allowableChildTMAIndices))


    """
    Add a new BeliefNode
    Inputs:
      beliefNodeIndex: Index of belief node, 1-numBeliefNodes+1
      beliefNodeDescriptor: String describing belief node  (e.g., B1)
      beliefNodexE: Observation associated with this belief node
    """
    def appendToBeliefNodes(self, beliefNodeIndex, beliefNodeDescriptor, beliefNodexE):
        self.beliefNodes.append(BeliefNode(beliefNodeIndex, beliefNodeDescriptor, beliefNodexE))

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
        value = 0
        maxTime = 50

        currentTime = 0
        numPackagesDelivered = 0

        while currentTime < maxTime and numPackagesDelivered < numPackagesGoal:
            # First check completion of TMAs and update xE accordingly
            for agent in self.agents:
                agent.currentTMACountdown -= 1
                if agent.currentTMACountdown <= 0:  # TMA completed
                    # Update belief node. Check for TMA failure; if so, don't update belief
                    agent.updateBeliefNode(self.beliefNodes)
                    if agent.currentBeliefNode.index in [1,2]: #At base 1 or 2
                        agent.setNodeXe(agent.currentTMAIndex, 1, 0, isOutputOn)
                        if isOutputOn:
                            print('Agent ', agent.index, 'added to belief node ', agent.currentBeliefNode.index)

                    if agent.packagePsi and ((agent.currentTMAIndex == 10 and agent.packagePsi == 1) or (agent.currentTMAIndex == 11 and agent.packagePsi == 2)):
                        # Package delivery, agent has package and is either delivering or joint delivering
                        if agent.packageDelta == agent.currentBeliefNode.index - 3:  # Make sure we're delivering to the right destination
                            value += (gamma**currentTime) * self.TMAs[agent.currentTMAIndex].rewards[agent.currentBeliefNode.index]
                            numPackagesDelivered += 1

                        agent.packageDelta, agent.packagePsi = (0, 0)

                    if isOutputOn:
                        print("Agent", agent.index, " finished its TMA and is at belief node ", agent.currentBeliefNode.descriptor)

            jointPickupTMAChosen = 0
            jointActionChosen = 0
            fellowAgentTMAIdx = 0
            fellowAgentTMATimer = 0
            fellowAgentDelta = 0
            fellowAgentPsi = 0

            # Now assign next TMAs based on new xE
            for agent in self.agents:
                if agent.currentTMACountdown <= 0:  # TMA complete
                    if jointPickupTMAChosen:
                        newTMAIndex = 9
                        agent.currentTMACountdown = fellowAgentTMATimer
                        agent.packageDelta = fellowAgentDelta
                        agent.packagePsi = fellowAgentPsi
                        if isOutputOn:
                            print("Agent ", agent.index, " forced to choose joint pickup TMA 9 (",self.TMAs[8].descriptor,
                                  " timer = ", agent.currentTMACountdown, "). It is currently at belief node ",agent.currentBeliefNode.descriptor)
                    elif jointActionChosen:
                        newTMAIndex = fellowAgentTMAIdx
                        agent.currentTMACountdown = fellowAgentTMATimer
                        if isOutputOn:
                            print("")
                    else:
                        if isOutputOn:
                            print("")
                        # ***CHECK INDICES HERE***
                        newTMAIndex = agent.executePolicy(agent.currentBeliefNode.envObs.getXeIndex(agent.index, agent.packageDelta, agent.currentTMAIndex, isOutputOn))
                        #Set timer for agent
                        agent.currentTMACountdown = self.TMAs[newTMAIndex-1].sampleTau(agent.currentBeliefNode)
                        if isOutputOn:
                            print("Agent ", agent.index, " sees xE ", agent.currentBeliefNode.index," and chooses TMA ", newTMAIndex)

                        #For pickup TMAs, save delivery location for next turn
                        if newTMAIndex in [8,9]:
                            agent.packageDelta = agent.currentBeliefNode.envObs.delta
                            agent.packagePsi = agent.currentBeliefNode.envObs.psi

                    #For goto *somewhere* tasks, need to remove self from availability
                    if newTMAIndex != 13:
                        if agent.currentBeliefNode.index in [1, 2]:
                            # Fail if trying to joint pickup a small package or solo pickup large
                            if (agent.currentBeliefNode.envObs.psi == 1 and newTMAIndex == 9) or (agent.currentBeliefNode.envObs.psi == 2 and newTMAIndex == 8):
                                agent.currentTMACountdown = -100
                            else:
                                agent.setNodeXe(agent.currentTMAIndex, 0, 1, isOutputOn)

                            if isOutputOn:
                                print('Agent ', agent.index, ' removed itself from the current node')

                    if newTMAIndex == 9:  # Joint pickup needs to coordinate
                        if jointPickupTMAChosen:  # Other agent initiated joint pickup
                            if agent.currentBeliefNode in [1, 2]:
                                agent.setNodeXe(newTMAIndex, 0, 1, isOutputOn)

                            jointPickupTMAChosen, fellowAgentTMATimer, fellowAgentDelta, fellowAgentDelta = (0, 0, 0, 0)
                        else:  # You are first agent to intiate, tell friend to help
                            jointPickupTMAChosen = 1
                            fellowAgentTMATimer = agent.currentTMACountdown
                            fellowAgentDelta = agent.packageDelta
                            fellowAgentPsi = agent.packagePsi

                    elif newTMAIndex in [6, 7, 11]: #Joint goto d1/d2, joint put down
                        if jointActionChosen:
                            jointActionChosen, fellowAgentTMATimer, fellowAgentTMAIdx = (0, 0, 0)
                        else:
                            jointActionChosen = 1  # Flag other agent for help
                            fellowAgentTMATimer = agent.currentTMACountdown
                            fellowAgentTMAIdx = newTMAIndex

                    # If new TMA is not a valid child of previous, fail
                    if newTMAIndex not in self.TMAs[agent.currentTMAIndex].allowableChildTMAIndices:
                        agent.currentTMACountdown = -100

                    agent.currentTMAIndex = newTMAIndex

            currentTime += 1

        if numPackagesDelivered < numPackagesGoal:
            completionTime = -1
        else:
            completionTime = currentTime

        return value, completionTime

    """
    Print allowable TMAs from all TMA indices
    """
    def printAllowableTMAs(self):
        for TMAIndex in range(1, self.numTMAs+1):
            print(TMAIndex)
            print(self.TMAs[TMAIndex-1].allowableChildTMAIndices)

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
            else: # Otherwise, specific termination set for this TMA
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
        pass

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

        tempIndex = 1
        for node in rootNodes:
            rootNodeTMAIndices











