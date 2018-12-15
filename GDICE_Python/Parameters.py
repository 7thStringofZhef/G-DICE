# GDICE parameter object
# Inputs:
#   numNodes: N_n number of nodes for the FSC used by GDICE
#   numIterations: N_k number of iterations of GDICE to perform
#   numSamples: N_s number of samples to take for each iteration from each node
#   numSimulationsPerSample: Number of times to run the environment for each sampled controller. Values will be averaged over these runs
#   numBestSamples: N_b number of samples to keep from each set of samples
#   learningRate: 0-1 alpha value, learning rate at which controller shifts probabilities
#   valueThreshold: If not None, ignore all samples with worse values, even if that means there aren't numBestSamples
class GDICEParams(object):
    def __init__(self, numNodes=10, numIterations=30, numSamples=50, numSimulationsPerSample=1000, numBestSamples=5, learningRate=0.1, valueThreshold=None, timeHorizon=100):
        self.numNodes = numNodes
        self.numIterations = numIterations
        self.numSamples = numSamples
        self.numSimulationsPerSample = numSimulationsPerSample
        self.numBestSamples = numBestSamples
        self.learningRate = learningRate
        self.valueThreshold = valueThreshold
        self.timeHorizon = timeHorizon
        self.buildName()

    # Name for use in saving files
    def buildName(self):
        self.name = 'N' + str(self.numNodes) + '_K' + str(self.numIterations) + '_S' + str(self.numSamples) + '_sim' + \
                    str(self.numSimulationsPerSample) + '_B' + str(self.numBestSamples) + '_lr' + \
                    str(self.learningRate) + '_vT' + ('None' if self.valueThreshold is None else str(self.valueThreshold)) + \
                                                                                                '_tH' + str(self.timeHorizon)