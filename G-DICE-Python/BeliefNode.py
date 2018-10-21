from __future__ import absolute_import, print_function, division

class BeliefNode(object):

    """
    Constructor. Makes a BeliefNode object
    Inputs:
      index: Unique BeliefNode index
      name: String name for belief node
      xE: Environmental observation
    """
    def __init__(self, index, name, xE):
        self.index = index
        self.name = name
        self.envObs = xE

    """
    Run samplePackage on environmental observation
    """
    def xeSamplePackage(self):
        self.envObs.samplePackage()

    """
    Set phi for the environmental observation
    """
    def xeSetPhi(self, agentIndex, hasArrived, hasLeft):
        self.envObs.setPhi(agentIndex, hasArrived, hasLeft)
