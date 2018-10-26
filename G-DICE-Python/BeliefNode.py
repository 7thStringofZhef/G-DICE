from __future__ import absolute_import, print_function, division

"""
Class to represent a belief node in the belief space

In the package delivery scenario, this could be base 1, base 2, destination 1, destination 2, destination inside airspace

Instance Variables:
  index: Unique BeliefNode index
  descriptor: String describing this belief node
  xE: Environmental observation
"""
class BeliefNode(object):

    """
    Constructor.
    """
    def __init__(self, index, descriptor, xE):
        self.index = index
        self.descriptor = descriptor
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
