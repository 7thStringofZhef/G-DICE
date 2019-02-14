# GDICE parameter object
# Inputs:
#   numNodes: N_n number of nodes for the FSC used by GDICE. Can also be a list/tuple/ndarray for each controller
#   numIterations: N_k number of iterations of GDICE to perform
#   numSamples: N_s number of samples to take for each iteration from each node
#   numSimulationsPerSample: Number of times to run the environment for each sampled controller. Values will be averaged over these runs
#   numBestSamples: N_b number of samples to keep from each set of samples
#   learningRate: 0-1 alpha value, learning rate at which controller shifts probabilities
#   valueThreshold: If not None, ignore all samples with worse values, even if that means there aren't numBestSamples
#   timeHorizon: If not None, run each sampled policy on the environment for this number of timesteps
class GDICEParams(object):
    def __init__(self, numNodes=10, numIterations=30, numSamples=50, numSimulationsPerSample=1000, numBestSamples=5, learningRate=0.1, valueThreshold=None, timeHorizon=100, centralized=True):
        self.numNodes = numNodes
        self.numIterations = numIterations
        self.numSamples = numSamples
        self.numSimulationsPerSample = numSimulationsPerSample
        self.numBestSamples = numBestSamples
        self.learningRate = learningRate
        self.valueThreshold = valueThreshold
        self.timeHorizon = timeHorizon
        self.centralized = centralized
        self.buildName()

    # Name for use in saving files
    def buildName(self):
        # use this for DPOMDPs
        nodeStr = ''
        if isinstance(self.numNodes, (list, tuple)):
            nodeStr += '[' + '_'.join([str(n) for n in self.numNodes]) + ']'
        else:
            nodeStr += str(self.numNodes)

        self.name = 'N' + nodeStr + '_K' + str(self.numIterations) + '_S' + str(self.numSamples) + '_sim' + \
                    str(self.numSimulationsPerSample) + '_B' + str(self.numBestSamples) + '_lr' + \
                    str(self.learningRate) + '_vT' + ('None' if self.valueThreshold is None else str(self.valueThreshold)) + \
                                                                                                '_tH' + str(self.timeHorizon)
        # Prepend if using decentralized controllers
        if not self.centralized:
            self.name = 'De_' + self.name

    # Parse a string for parameters, set self to those parameters
    # String must be the name from above
    def fromName(self, name):
        self.centralized = False if name.startswith('De_') else True
        self.numNodes = int((name.split('N'))[1].split('_K')[0])
        self.numSamples = int((name.split('_K'))[1].split('_S')[0])
        self.numSimulationsPerSample = int((name.split('_S'))[1].split('_B')[0])
        self.numBestSamples = int((name.split('_B'))[1].split('_lr')[0])
        self.learningRate = float((name.split('_lr'))[1].split('_vT')[0])
        self.valueThreshold = None if (name.split('_vT'))[1].split('_tH')[0] == 'None' else float((name.split('_lr'))[1].split('_vT')[0])
        self.timeHorizon = int((name.split('_tH'))[1])
        self.buildName()

