#!/usr/bin/env python

import numpy as np

from Controller_Node import ControllerNode

class GraphPolicyController(object):
    
    def __init__(self, N_MA, N_OB, N_s, N_n, alpha):      
        self.nodes = list()         # nodes in the graph policy controller
        self.N_MA = N_MA            # number of actions or macro-actions
        self.N_OB = N_OB            # number of observations or macro-observations
        self.N_s = N_s              # number of samples
        self.N_n = N_n              # number of nodes in the graph policy controller
        self.alpha = alpha          # learning rate
        
        # initialize nodes in the graph policy controller
        for node_id in range(self.N_n):
            new_node = ControllerNode(node_id, self.N_MA, self.N_OB, self.N_s, self.N_n)
            self.nodes.append(new_node)
    
    def sample(self):
        # sample actions/macro-actions for each node
        for node_id in range(self.N_n):
            self.nodes[node_id].sampleMAs()
            
            # sample transitions for each node given different observations
            for ob_id in range(self.N_OB):
                self.nodes[node_id].sampleTrans(ob_id)
    
    def update_sample_distr(self, N_b_policies):
        #update the sampleing distribution for each node
        if len(N_b_policies) > 0:
            for node_id in range(self.N_n):
                # weight is used to learn the new distribution each step automatically
                weight = 1.0 / len(N_b_policies)
                # based on learning equation to update 
                new_SDMA = self.nodes[node_id].SDMA * (1 - self.alpha) 
                new_SDOB = self.nodes[node_id].SDOB * (1 - self.alpha)
                
                for s_id in N_b_policies:
                    # update the sampling distribution over actions/macro-actions for each node
                    MA_sample = self.nodes[node_id].sampledMAs[s_id]
                    new_SDMA[MA_sample] += weight * self.alpha
                    
                    for ob_id in range(self.N_OB):
                        # updeate the sampling distribution of trainsitions for each node 
                        NextNode_sample = self.nodes[node_id].transitions[s_id][ob_id]
                        new_SDOB[NextNode_sample][ob_id] += weight * self.alpha
                
                self.nodes[node_id].SDMA = new_SDMA
                self.nodes[node_id].SDOB = new_SDOB
                
    def setupPolicy(self, s_id):
        for node_id in range(self.N_n):
            self.nodes[node_id].setupMAandTrans(s_id)
            
    def getNextNodeandMA(self, cur_node, cur_ob):
        # based on current node and received observation, move to next node
        next_node_id = self.nodes[cur_node].curTrans[cur_ob]
        next_MA_id = self.nodes[next_node_id].curMA
        return next_node_id, next_MA_id
    
    def getCurPolicy(self):
        MAs = np.zeros(self.N_n, dtype = int)
        Trans = np.zeros((self.N_n, self.N_OB), dtype = int)
        for node_id in range(self.N_n):
            MAs[node_id] = self.nodes[node_id].curMA
            Trans[node_id, :] = self.nodes[node_id].curTrans
        return MAs, Trans
    
    def getPolicy(self, s_id):
        MAs = np.zeros(self.N_n, dtype = int)
        Trans = np.zeros((self.N_n, self.N_OB), dtype = int)
        for node_id in range(self.N_n):
            MAs[node_id] = self.nodes[node_id].sampledMAs[s_id]
            Trans[node_id, :] = self.nodes[node_id].transitions[s_id, :]
        return MAs, Trans
        
         
            