from __future__ import absolute_import, print_function

import numpy as np
from numpy import nan

from .Base import DomainConfiguration
from ..BeliefNode import BeliefNode
from ..TMA import TMA
from ..EnvObs import EnvObs

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


class PackageDeliveryDomain(DomainConfiguration):

    """
    Constructor
    """
    def __init__(self):
        super(PackageDeliveryDomain, self).__init__()
        self.envObsToBeliefNodeDict = envObsBeliefNodeDictForPackageDelivery
        self.tmaDescriptors = tmaDescriptorsForPackageDelivery
        self.tmaTau = tmaTauForPackageDelivery
        self.tmaTerminalBeliefs = tmaBTermForPackageDelivery
        self.tmaRewards = tmaRewardsForPackageDelivery
        self.tmaChildIndices = tmaChildTMAsForPackageDelivery
        self.tmaDict = tmaDictForPackageDelivery
        self.validTMATransitions = validTMATransitions