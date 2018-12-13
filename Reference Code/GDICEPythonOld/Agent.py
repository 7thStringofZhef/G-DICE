from __future__ import absolute_import, print_function, division

from .Policy import Policy, GraphPolicyController
"""
Class to represent an agent acting in the environment

Instance Variables:
  index: Agent's unique index. Starts at 1. May make sense to switch to uuid
  currentBeliefNode: Current belief node 
  currentTMAIndex: Currently executing task macro action. 0 if no TMA
  packageDelta: 0 if no package, otherwise destination of package 
  packagePsi: 0 if no package, otherwise size of package
  policy: Policy controller
  currentPolicyNodeIndex: Agent's position in policy controller graph
  currentTMACountdown: Timer of TMA counts down until it's complete
"""

#Dict to map currentTMAIndex to belief node
beliefNodeUpdateDict = {
  **dict.fromkeys([5, 7], 5),  # Both goto D2 end at D2
  **dict.fromkeys([4, 6], 4),  # Both goto D1 end at D1
  1: 1,  # Goto B1 ends at B1
  2: 2,  # Goto B2 ends at B2
  3: 3   # Goto B3 ends at B3
}
class Agent(object):

    """
    Constructor
    """
    def __init__(self, index, initialBeliefNode, initialTMAIndex):
        self.index = index
        self.currentBeliefNode = initialBeliefNode
        self.currentTMAIndex = initialTMAIndex
        self.packageDelta, self.packagePsi = (0, 0)  # No package
        self.policy = None  # No policy yet
        self.currentPolicyNodeIndex = 0  # No policy yet
        self.currentTMACountdown = 1

    """
    Updates Agent's current belief node depending on TMA index
    Inputs:
      beliefNodes: A list of belief nodes from which to draw
      *currentTMAIndex: Implicit
    """
    def updateBeliefNode(self, beliefNodes):
        try:
            self.currentBeliefNode = beliefNodes[beliefNodeUpdateDict[self.currentTMAIndex]]
        except:
            return  # In all other cases, belief node remains the same

    """
    Set the environmental observation xe for the current belief node
    Inputs:
      TMAIndex: index of TMA
      hasArrivedOrIsComplete: Has agent finished this TMA/arrived at destination?
      hasLeftOrJustStarted: Has agent just left destination or just started TMA
      isOutputOn: Print debug output
      
    """
    def setNodeXe(self, TMAIndex, hasArrivedOrIsComplete, hasLeftOrJustStarted, isOutputOn=False):
        if hasArrivedOrIsComplete == hasLeftOrJustStarted:
            raise ValueError('hasArrived cannot be equal to hasLeft, no zero-time TMAs')

        #Only change xE at node B1 or B2. Doesn't matter otherwise in this domain, even at rendezvous
        if self.currentBeliefNode.index in [1, 2]:  # B1,B2
            if TMAIndex in [8, 9]:  # Pickup/Joint pickup
                self.currentBeliefNode.xeSamplePackage()  # Generate a new package
                self.currentBeliefNode.xeSetPhi(self.index, 0, 1)  # Picking up, so not available
                if isOutputOn:
                    print('Agent ',self.index, ' has removed itself from current node')
            else:  # Not picking up
                if hasArrivedOrIsComplete:
                    self.currentBeliefNode.xeSetPhi(self.index, 1, 0)
                elif hasLeftOrJustStarted:
                    self.currentBeliefNode.xeSetPhi(self.index, 0, 1)
        else:
            raise ValueError('You should not be updating XE from belief node ',self.currentBeliefNode.index,' (',self.currentBeliefNode.descriptor,')')

    """
    Execute the agent's policy, move to next TMA
    
    Input:
      currentXeIndex: Current environmental observation's domain index
    Output:
      nextTMAIndex: Index of next TMA according to policy
    """
    def executePolicy(self, currentXeIndex):
        self.currentPolicyNodeIndex, nextTMAIndex = self.policy.getNextTMAIndex(self.currentPolicyNodeIndex, currentXeIndex)
        return nextTMAIndex

    """
    Set the agent's policy
    
    Input:
      policy: A Policy or GraphPolicyController instance. If neither, this method throws an error
    """
    def setPolicy(self, policy):
        if isinstance(policy, Policy) or isinstance(policy, GraphPolicyController):
            self.policy = policy
        else:
            raise ValueError('Not a valid policy!')






