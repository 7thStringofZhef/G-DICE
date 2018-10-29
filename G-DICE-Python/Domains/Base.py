from __future__ import absolute_import, print_function


"""
Class to hold domain-specific dictionaries
"""
class DomainConfiguration(object):
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

