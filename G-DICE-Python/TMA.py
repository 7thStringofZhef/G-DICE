from __future__ import absolute_import, print_function, division

import numpy.random as random
import numpy as np

"""
Class to represent a task macro-action

TMAs can apply to one or more agents
"""
class TMA(object):
    tauStdDeviation = 0.3 #Normalized standard deviation of completion time

    def __init__(self, index, descriptor, tau, terminalBeliefs, rewards, allowableChildTMAIndices):
        self.index = index #Index of this TMA within domain
        self.descriptor = descriptor #String describing TMA
        self.tau = tau #List of expected completion time (in timesteps) from belief node
        self.terminalBeliefs = terminalBeliefs #List of belief nodes for which this TMA is considered completed. NaN if same, 0 if not terminated, 1 if terminated
        self.rewards = rewards #Rewards for each belief node
        self.allowableChildTMAIndices = allowableChildTMAIndices #List of TMAs that can be initiated from this TMA's terminal belief node set

    #Sample a completion time for this TMA using the given std deviation
    #Input: currentBeliefNode
    #Output: completion time
    def sampleTau(self, currentBeliefNode):
        if self.index == 13: #Wait TMA waits deterministically
            return self.tau(currentBeliefNode.index)
        sample = self.tau(currentBeliefNode.index) * (1 + self.tauStDevParam * random.randn())

        #If it would round to 0, and tau is not 0, return tau
        if (sample < 0.5 and self.tau(currentBeliefNode.index)!=0):
            return self.tau(currentBeliefNode.index)

        #If our tau was NaN, we shouldn't be able to do this, return 0. Throw error?
        if np.isnan(sample):
            raise ValueError('Tried to sample from a NaN tau! Are you sure you can call TMA #%i (%s) from the passed-in belief node (%s)?', self.index, self.descriptor, currentBeliefNode.descriptor)
            return 0

        return np.round(sample)

