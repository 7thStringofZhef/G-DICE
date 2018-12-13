#!/usr/bin/env python

class Agent(object):
    
    def __init__(self, ag_id):
        self.ag_id = ag_id
        self.currentNode_id = 0
        
    def getPolicy(self, curPolicy):
        self.curPolicy = curPolicy
        self.curMA = self.curPolicy.nodes[self.currentNode_id].curMA
        self.curTrans = self.curPolicy.nodes[self.currentNode_id].curTrans
    
    def setupState(self, state_id):
        self.curstate = state_id
        
    def setupNode(self, cur_obs):
        self.currentNode_id, self.curMA = self.curPolicy.getNextNodeandMA(self.currentNode_id, cur_obs)
        
    def resetag(self):
        self.currentNode_id = 0
        self.curMA = self.curPolicy.nodes[self.currentNode_id].curMA
        self.curTrans = self.curPolicy.nodes[self.currentNode_id].curTrans
        