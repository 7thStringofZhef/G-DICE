from __future__ import absolute_import, print_function

import numpy as np
from numpy import nan

from ..Base import Domain
from .EnvObs import PackageDeliveryEnvObs as EnvObs
from ...BeliefNode import BeliefNode
from ...TMA import TMA
from ...Agent import Agent

# Mapping of environmental observation index to BeliefNode for PackageDelivery
# Base nodes always have a package
# Delivery nodes do not
envObsBeliefNodeDictForPackageDelivery = {
    1: BeliefNode(1, 'B1', EnvObs([1, 2], [1, 2, 3])),
    2: BeliefNode(2, 'B2', EnvObs([1, 2], [1, 2, 3])),
    3: BeliefNode(3, 'R', EnvObs([], [])),
    4: BeliefNode(4, 'D1', EnvObs([], [])),
    5: BeliefNode(5, 'D2', EnvObs([], []))
}

# TMA descriptors for package delivery
tmaDescriptorsForPackageDelivery = [
    'Go to B1',
    'Go to B2',
    'Go to R',
    'Go to D1',
    'Go to D2',
    'Joint go to D1',
    'Joint go to D2',
    'Pick up package',
    'Joint pick up package',
    'Put down package',
    'Joint put down package',
    'Place package on truck',
    'Wait'
]

# TMA tau for package delivery
tmaTauForPackageDelivery = [
    np.array([nan, 2, 4, 3, 2]),
    np.array([2, nan, 4, 2, 4]),
    np.array([3, 4, nan, nan, nan]),
    np.array([3, 2, nan, nan, nan]),
    np.array([2, 4, nan, nan, nan]),  # End of single gotos
    np.array([3, 2, nan, nan, nan])*1.5,
    np.array([2, 4, nan, nan, nan])*1.5,  #End of joint gotos
    np.array([1, 1, nan, nan, nan]),
    np.array([1, 1, nan, nan, nan])*1.5,  # End of pickup
    np.array([nan, nan, nan, 1, 1]),
    np.array([nan, nan, nan, 1, 1])*1.5,  # End of delivery
    np.array([nan, nan, 1, nan, nan]),
    np.array([nan, nan, nan, nan, nan])
]

# TMA terminal beliefs for package delivery
tmaBTermForPackageDelivery = [
    np.array([1, 0, 0, 0, 0]),
    np.array([0, 1, 0, 0, 0]),
    np.array([0, 0, 1, 0, 0]),
    np.array([0, 0, 0, 1, 0]),
    np.array([0, 0, 0, 0, 1]),
    np.array([0, 0, 0, 1, 0]),
    np.array([0, 0, 0, 0, 1]),  # end of goto tmas
    np.array([nan, nan, nan, nan, nan]),
    np.array([nan, nan, nan, nan, nan]),
    np.array([nan, nan, nan, nan, nan]),
    np.array([nan, nan, nan, nan, nan]),
    np.array([nan, nan, nan, nan, nan]),
    np.array([nan, nan, nan, nan, nan])
]

# TMA rewards for package delivery
tmaRewardsForPackageDelivery = [
    np.zeros(5),
    np.zeros(5),
    np.zeros(5),
    np.zeros(5),
    np.zeros(5),
    np.zeros(5),
    np.zeros(5),  # End of goto tmas
    np.array([0, 0, nan, nan, nan]),
    np.array([0, 0, nan, nan, nan]),  # End of pickup tmas
    np.array([nan, nan, nan, 1, 1]),
    np.array([nan, nan, nan, 3, 3]),  # End of deliver tmas
    np.array([nan, nan, 0, nan, nan]),
    np.zeros(5)
]

# TMA allowable child TMA indices for package delivery
# Must be int type to be indices
tmaChildTMAsForPackageDelivery = [
    np.array([2, 3, 8, 9], dtype=np.int16),
    np.array([1, 3, 8, 9], dtype=np.int16),
    np.array([1, 2, 12], dtype=np.int16),
    np.array([1, 2, 10], dtype=np.int16),
    np.array([1, 2, 10], dtype=np.int16),  # end of single goto
    np.array([11], dtype=np.int16),
    np.array([11], dtype=np.int16),  # end of goto tmas
    np.array([3, 4, 5], dtype=np.int16),
    np.array([6, 7], dtype=np.int16),  # end of pickup tmas
    np.array([1, 2], dtype=np.int16),
    np.array([1, 2], dtype=np.int16),
    np.array([1, 2], dtype=np.int16),
    np.array([13], dtype=np.int16)
]

# TMA definitions for package delivery
tmaDictForPackageDelivery = {k+1: TMA(k+1, tmaDescriptorsForPackageDelivery[k], tmaTauForPackageDelivery[k],
                                      tmaBTermForPackageDelivery[k], tmaRewardsForPackageDelivery[k],
                                      tmaRewardsForPackageDelivery[k]) for k in range(len(tmaDescriptorsForPackageDelivery))}


                                #  1   2   3   4   5   6   7   8   9  10  11  12  13
                                #000 110 120 130 210 220 230 111 121 131 211 221 231
validTMATransitions = np.array([[-1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  #1 -goto b1 (at b1)
                                [-1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 2 -goto b2 (at b2)
                                [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # 3 -goto r (at r)
                                [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # 4 -goto d1 (at d1)
                                [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # 5 -goto d2 (at d2)
                                [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # 6 -joint goto d1 (at d1)
                                [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # 7 -joint goto d2 (at d2)
                                [-1, 0,  0,  0, -1, -1, -1,  0,  0,  0, -1, -1, -1],  # 8 -pickup pkg (at b1 or b2) %this is a special case where the observations MUST COME BEFOREA NEW PKG IS GENERATED!!
                                [-1,-1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1],  # 9 -joint pickup pkg (at b1 or b2)
                                [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # 10 -put down pkg (at d1 or d2)
                                [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # 11 -joint put down pkg (at d1 or d2)
                                [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # 12 -place on truck (at r)
                                [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])  # 13 -wait (anywhere)


class PackageDeliveryDomain(Domain):

    """
    Constructor.
    """
    def __init__(self):
        super(PackageDeliveryDomain, self).__init__()
        self.beliefNodes = [] #List of belief nodes in this domain
        self.TMAs = [] #List of task macro-actions in this domain
        self.numTMAs = 0 #Initially no TMAs
        self.numBeliefNodes = 5 #Number of expected belief nodes
        self.agents = [] #List of agents in domain

        # Define belief nodes
        for beliefNodeIndex in range(1, len(envObsBeliefNodeDictForPackageDelivery) + 1):
            self.appendToBeliefNodes(envObsBeliefNodeDictForPackageDelivery[beliefNodeIndex])

        # Define agents. Start both at B1
        self.appendToAgents(Agent(1, self.beliefNodes[0], 1))
        self.appendToAgents(Agent(2, self.beliefNodes[0], 1))

        # Define TMAs
        for tmaIndex in range(1, len(tmaDictForPackageDelivery)+1):
            self.appendToTMAs(tmaDictForPackageDelivery[tmaIndex])
        self.numTMAs = len(self.TMAs)

        # Init environmental state
        self.initXe()

        # [obj.psi obj.delta obj.phi] = [size, destination, other agents avail]


    """
    Set agents in xE, init packages at bases
    """
    def initXe(self):
        for agent in self.agents:
            agent.currentBeliefNode.xeSetPhi(agent.agentIndex, 1, 0)
        for baseIndex in [1, 2]:
            self.beliefNodes[baseIndex-1].xeSamplePackage()

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

