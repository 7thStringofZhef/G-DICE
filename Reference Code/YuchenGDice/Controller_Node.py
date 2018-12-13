#!/usr/bin/env python

import numpy as np

class ControllerNode(object):
    
    def __init__(self, node_id, N_MA, N_OB, N_s, N_n):
        
        self.node_id = node_id              # id of node
        self.N_MA = N_MA                    # number of actions or macro-actions
        self.N_OB = N_OB                    # number of observations or macro-observations
        self.N_s = N_s                      # number of samples
        self.N_n = N_n                      # number of nodes
        self.transitions = np.zeros((self.N_s, self.N_OB), dtype = int)  # node's transitions of all samples
        self.SDMA = np.ones(self.N_MA) / self.N_MA                       # sampling distribution over actions/macro-actions
        self.SDOB = np.ones((self.N_n, self.N_OB)) / self.N_n            # sampling distribution over all nodes given observation
        
    def sampleMAs(self):       
        self.sampledMAs = np.random.choice(range(self.N_MA), self.N_s, p=self.SDMA)
               
    def sampleTrans(self, ob_id):
        self.transitions[:, ob_id] = np.random.choice(range(self.N_n), self.N_s, p=self.SDOB[:, ob_id])
        
    def setupMAandTrans(self, s_id):
        self.curMA = self.sampledMAs[s_id]
        self.curTrans = self.transitions[s_id, :]
        
    def set_SDMA(self, new_SDMA):
        self.SDMA = new_SDMA
        