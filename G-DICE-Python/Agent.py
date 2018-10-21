from __future__ import absolute_import, print_function, division


class Agent(object):

    """
    Constructor. Creates an Agent instance
    Inputs:
      index: Agent's unique index. Should I replace with uuid?
      initialBeliefNode: Starting belief node
      initialTMAIndex: Macro-action index
    Outputs:
      Agent: An instantiation of an agent
    """
    def __init__(self, index, initialBeliefNode, initialTMAIndex):
        self.index = index
        self.currentBeliefNode = initialBeliefNode
        self.currentTMAIndex = initialTMAIndex

    """
    Updates Agent's current belief node depending on current TMA index
    Inputs:
      beliefNodes: A list of belief nodes from which to draw
      *currentTMAIndex: Implicit
    """
    def updateBeliefNode(self, beliefNodes):
        if self.currentTMACountdown<=-1:
            return
        else:
            if self.currentTMAIndex<=4:
                self.currentBeliefNode = beliefNodes[self.currentTMAIndex]
            elif self.currentTMAIndex in [5,7]:
                self.currentBeliefNode = beliefNodes[5]
            elif self.currentTMAIndex == 6:
                self.currentBeliefNode = beliefNodes[4]

    """
    Inputs:
      indexTMA:
      hasArrivedOrIsComplete:
      hasLeftOrJustStarted
      isOutputOn:
      
    """
    def setNodeXe(self, indexTMA, hasArrivedOrIsComplete, hasLeftOrJustStarted, isOutputOn):
        if hasArrivedOrIsComplete==hasLeftOrJustStarted:
            raise ValueError('hasArrived cannot be equal to hasLeft, no zero-time TMAs')






