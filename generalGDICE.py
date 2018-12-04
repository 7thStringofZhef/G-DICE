from gym_pomdps import list_pomdps, POMDP
import gym
import numpy as np
import numpy.random as npr
from multiprocessing import Pool

class FiniteStateController(object):
    def __init__(self, numNodes, numActions, numObservations):
        self.numNodes = numNodes
        self.numActions = numActions
        self.numObservations = numObservations
        self.currentNode = None
        self.initActionNodeProbabilityTable()
        self.initObservationNodeTransitionProbabilityTable()

    # Probability of each action given being in a certain node
    def initActionNodeProbabilityTable(self):
        initialProbability = 1 / self.numActions
        self.actionProbabilities = np.full((self.numNodes, self.numActions), initialProbability)

    # Probability of transition from 1 node to second node given obsersvation
    def initObservationNodeTransitionProbabilityTable(self):
        initialProbability = 1 / self.numNodes
        self.nodeTransitionProbabilities = np.full((self.numNodes,
                                                    self.numNodes,
                                                    self.numObservations), initialProbability)

    # Set the current node of the controller
    def setNode(self, nodeIndex):
        self.currentNode = nodeIndex

    # Returns the index of the current node
    def getCurrentNode(self):
        return self.currentNode

    # Reset the controller to default probabilities
    def reset(self):
        self.setNode(None)
        self.initActionNodeProbabilityTable()
        self.initObservationNodeTransitionProbabilityTable()

    # Get an action using the current node according to probability. Can sample multiple actions
    def sampleAction(self, numSamples=1):
        return npr.choice(np.arange(self.numActions), size=numSamples, p=self.actionProbabilities[self.currentNode, :])

    # Get an action from all nodes according to probability. Can sample multiple actions
    # Outputs numNodes * numSamples
    def sampleActionFromAllNodes(self, numSamples=1):
        actionIndices = np.arange(self.numActions)
        return np.array([npr.choice(actionIndices, size=numSamples, p=self.actionProbabilities[nodeIndex,:])
                         for nodeIndex in range(self.numNodes)], dtype=np.int32)

    # Get the next node according to probability given current node and observation index
    # Can sample multiple transitions
    # DOES NOT set the current node
    def sampleObservationTransition(self, observationIndex, numSamples=1):
        return npr.choice(np.arange(self.numNodes), size=numSamples, p=self.nodeTransitionProbabilities[self.currentNode, :, observationIndex])

    # Get the next node for each node given observation index
    # Outputs numNodes * numSamples
    def sampleObservationTransitionFromAllNodes(self, observationIndex, numSamples=1):
        nodeIndices = np.arange(self.numNodes)
        return np.array([npr.choice(nodeIndices, size=numSamples, p=self.nodeTransitionProbabilities[nodeIndex, :, observationIndex])
                         for nodeIndex in range(self.numNodes)], dtype=np.int32)

    # Get the next node for all nodes for all observation indices
    # Outputs numObs * numNodes * numSamples
    def sampleAllObservationTransitionsFromAllNodes(self, numSamples=1):
        obsIndices = np.arange(self.numObservations)
        return np.array([self.sampleObservationTransitionFromAllNodes(obsIndex, numSamples)
                         for obsIndex in obsIndices], dtype=np.int32)


    # Update the probability of taking an action in a particular node
    # Can be used for multiple inputs if numNodeIndices = n, numActionIndices = m, and newProbability = n*m or a scalar
    def updateActionProbability(self, nodeIndex, actionIndex, newProbability):
        self.actionProbabilities[nodeIndex, actionIndex] = newProbability

    # Update the probability of transitioning from one node to a second given an observation
    def updateTransitionProbability(self, firstNodeIndex, secondNodeIndex, observationIndex, newProbability):
        self.nodeTransitionProbabilities[firstNodeIndex, secondNodeIndex, observationIndex] = newProbability

    # Get the current probability tables
    def save(self):
        return self.actionProbabilities, self.nodeTransitionProbabilities


# Evaluate controller(s) on an environment, given
# Inputs:
#   env: Gym-like environment to evaluate on
#   controller: A controller or list of controllers corresponding to agents in the environment
#   params: GDICEParams object
#   timeHorizon: Number of timesteps to evaluate to. If None, run each sample until episode is finished
#   parallel: Attempt to use python multiprocessing across samples. If not None, should be a Pool object

def evaluateFSCOnEnvironment(env, controller, params, timeHorizon=50, parallel=None):
    # Ensure controller matches environment
    assert env.action_space.n == controller.numActions
    assert env.state_space.n == controller.numObservations

    # Reset controller
    controller.reset()

    # Start variables
    bestActionProbs = None
    bestNodeTransitionProbs = None
    bestValue = np.NINF

    # Wrap the environment to sample multiple trajectories simulatenously
    multiEnv = MultiActionPOMDP(env, numTrajectories=params.numSamples)
    for iteration in params.numIterations:
        for timestep in timeHorizon:
            # For each node in controller, sample actions
            sampledActions = controller.sampleActionFromAllNodes(params.numSamples)

            # For each node, observation in controller, sample next node
            sampledNodes = controller.sampleAllObservationTransitionsFromAllNodes(params.numSamples)

            # For each sampled action, evaluate in environment
            obs, rewards, isDones = multiEnv.step(sampledActions)


    pass

# Evaluate a single sample
def evaluateSample(env, controller):
    pass



# GDICE parameter object
# Inputs:
#   numIterations: N_k number of iterations of GDICE to perform
#   numSamples: N_s number of samples to take for each iteration from each node
#   numBestSamples: N_b number of samples to keep from each set of samples
#   leareningRate: 0-1 alpha value, learning rate at which controller shifts probabilities
#   valueThreshold: If not None, ignore all samples with worse values, even if that means there aren't numBestSamples
class GDICEParams(object):
    def __init__(self, numIterations=30, numSamples=50, numBestSamples=5, learningRate=0.1, valueThreshold=None):
        self.numIterations = numIterations
        self.numSamples = numSamples
        self.numBestSamples = numBestSamples
        self.learningRate = learningRate
        self.valueThreshold = valueThreshold


# States, observations, rewards, actions, dones are now lists or np arrays
class MultiActionPOMDP(gym.Wrapper):

    def __init__(self, env, numTrajectories):
        assert isinstance(env, POMDP)
        super().__init__(env)
        self.numTrajectories = numTrajectories
        self.reset()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self):
        if self.env.start is None:
            self.state = self.np_random.randint(
                self.state_space.n, size=self.numTrajectories)
        else:
            self.state = self.np_random.multinomial(
                1, self.env.start, size=self.numTrajectories).argmax(1)

    # Step given an nparray or list of actions
    # If actions is a scalar, applies to all
    def step(self, actions):
        if np.isscalar(actions):
            actions = np.full(self.numTrajectories, actions, dtype=np.int32)

        newStates = np.array([self.np_random.multinomial(1, p).argmax() for p in self.env.T[self.state, actions]])
        obs =np.array([self.np_random.multinomial(1, p).argmax() for p in self.env.O[self.state, actions, newStates]])
        rewards = np.array(self.env.R[self.state, actions, newStates, obs])
        done = np.zeros(self.numTrajectories, dtype=bool)
        if self.env.episodic:
            done = self.env.D[self.state, actions]
            self.state = None
        else:
            self.state = newStates
        return obs, rewards, done, {}


if __name__ == "__main__":
    env = gym.make(list_pomdps()[2])  # 4x3-v0 POMDP
    multiEnv = MultiActionPOMDP(env, 50)
    multiEnv.step(0)
    controller = FiniteStateController(10, 4, 11)
    pool = Pool()  # Use a pool for parallel processing. Max # threads
