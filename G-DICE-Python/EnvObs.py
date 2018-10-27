from __future__ import absolute_import, print_function, division

import numpy as np

"""
Class to represent an environmental observation

An environmental observation is of the form x={psi, delta, phi}, where
  Psi: Package size {0,1,2}
  Delta: Package destination {1,2,3}
  Phi: Availability of nearby agents [List of agent indices]
   
Instance variables
  psi: Package size
  allPsis: Set of all psis. In package delivery, set of all package sizes. {1,2}
  delta: Package delivery destination
  allDeltas: Set of all deltas. In package delivery, set of all delivery destinations {1,2,3}
  phi: Array of indices of agents at the current belief node
    
"""

#Dict to map envObs to envObsIndex
packageDeliveryEnvObsIndexDict = {
    [0, 0, 0]: 1,  # Null case
    [1, 1, 0]: 2,  # small, dest1, no other agent
    [1, 2, 0]: 3,  # small, dest2, no agent
    [1, 3, 0]: 4,  # small, destr, no agent
    [2, 1, 0]: 5,  # large, dest1, no agent
    [2, 2, 0]: 6,  # large, dest2, no agent
    [2, 3, 0]: 7,  # large, destr, no agent
    [1, 1, 1]: 8,  # small, dest1, agent available
    [1, 2, 1]: 9,  # small, dest2, agent
    [1, 3, 1]: 10,  # small, destr, agent
    [2, 1, 1]: 11,  # large, dest1, agent
    [2, 2, 1]: 12,  # large, dest2, agent
    [2, 3, 1]: 13  # large, destr, agent
}

#Special case dict to map [TMAIndex, packageDelta] to envObsIndex
pickupIndexDict = {
    # Single pickup
    [8, 1]: 2,
    [8, 2]: 3,
    [8, 3]: 4,
    # Dual pickup
    [9, 1]: 11,
    [9, 2]: 12
}

class EnvObs(object):

    """
    Constructor
    """
    def __init__(self, allPsis, allDeltas):
        self.allPsis = allPsis
        self.allDeltas = allDeltas
        #  Start with null case
        self.setXeNull()

    """
    Get the index of this environmental observation
    Inputs:
      callerAgentIndex: Index of agent that called this function
      callerAgentPackageDelta: Destination of package carried by caller agent {1,2,3}
      currentTMAIndex: Agent's currently active task macro-action {1-13}
      isOutputOn: Print caller agent
    Output:
      xeIndex: Index of environmental observation as used by Domain. Starts at 1
    """
    def getXeIndex(self, callerAgentIndex, callerAgentPackageDelta, currentTMAIndex, isOutputOn=False):
        xe = [self.psi, self.delta, (not not self.phi)]
        if callerAgentIndex not in self.phi: #  No other agent present
            xe[2] = 0

        if isOutputOn:
            print('Agent ',callerAgentIndex,' called getXeIdx (TMA = ',currentTMAIndex,', package delta = ',callerAgentPackageDelta,'), xe = ',xe,', psi = ',self.psi,', delta = ',self.delta,', phi = ',self.phi)

        #  For pickup tasks, assign xeIndex based on delivery destination
        try:
            return pickupIndexDict[[currentTMAIndex, callerAgentPackageDelta]]
        except KeyError:
            pass

        try:
            return packageDeliveryEnvObsIndexDict[xe]
        except KeyError:
            raise ValueError('Received xe: ',xe,' which is impossible')

    """
    Clear environmental observation (set all to 0)
    """
    def setXeNull(self):
        self.psi, self.phi, self.delta = (0, [], 0)

    """
    Stochastically sample for a new package, using the list of all possible package sizes and destinations
    If the package is heading for rendezvous, it can only be small because there is only one ground robot
    
    Called after agent starts pickup of a package (if at B1, B2). 
    """
    def samplePackage(self):
        self.delta = np.random.choice(self.allDeltas)
        if self.delta == 3: #  If destined for rendezvous, only a small package is allowed
            self.psi = 1
        else:
            self.psi = np.random.choice(self.allPsis)

    """
    Called after an agent completes arrival and after an agent starts to leave (if at B1, B2)
    """
    def setPhi(self, agentIndex, hasArrived, hasLeft):
        if hasArrived and hasLeft:
            raise ValueError('Agent ',agentIndex, ' cannot simultaneously arrive at and leave a node')

        if hasArrived:
            self.phi.append(agentIndex)










