from __future__ import absolute_import, print_function, division

class EnvObs(object):

    """
    Instance variables
      psi: Package size
      allPsis: Set of all psis
      delta: Package delivery destination
      allDeltas: Set of all deltas
      phi: Array of indices of agents at the current belief node
    """

    """
    Constructor. Creates an EnvObs object
    Inputs:
      allPsis:
      allDeltas:
    Outputs:
      EnvObs object
    """
    def __init__(self, allPsis, allDeltas):
        self.allPsis = allPsis
        self.allDeltas = allDeltas

    """
    
    """
    def getXeIdx(self, callerAgentIndex, callerAgentPackageDelta, currentTMAIndex, isOutputOn):

