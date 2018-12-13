#!/usr/bin/env python

import numpy as np
from Agents import Agent
from Macro_Action import MA

class Domain(object):
    
    def __init__(self, N_state, N_agent, N_ma, N_obs, Discount):
        self.gamma = Discount # discount
        self.N_ma = N_ma  # number of MA
        self.N_obs = N_obs  # number of observation
        self.N_agent = N_agent  #number of agents
        self.N_state = N_state  # number of state
        
        # setup agent
        self.agents = list()         
        for ag_id in range(N_agent):
            new_agent = Agent(ag_id)
            self.agents.append(new_agent)
    
    def inputPolicyToAgents(self, PolicyControllers):
        for ag_id in range(self.N_agent):
            self.agents[ag_id].getPolicy(PolicyControllers[ag_id])
    
    def dpomdp_parser(self, path):
        fp = open(path)
        lines = fp.read().split("\n")
        fp.close()
        
        self.T_table = {}
        self.O_table = {}
        self.R_table = {}
        
        for i in range(len(lines)):
            if 'T:' in lines[i]:
                lines[i] = lines[i].split(':')
                if lines[i][1] not in self.T_table:
                    self.T_table[lines[i][1]] = np.zeros((self.N_state, self.N_state))
                self.T_table[lines[i][1]][int(lines[i][2])][int(lines[i][3])] = float(lines[i][4])
            
            elif 'O:' in lines[i]:            
                lines[i] = lines[i].split(':')
                lines[i][3] = lines[i][3].split(' ')
                if lines[i][1] not in self.O_table:                
                    self.O_table[lines[i][1]] = []
                    for state_id in range(self.N_state):
                        self.O_table[lines[i][1]].append(np.zeros(self.N_agent, dtype = int))
                for ag_id in range(self.N_agent):
                    self.O_table[lines[i][1]][int(lines[i][2])][ag_id] = int(lines[i][3][ag_id + 1])
                
            elif 'R:' in lines[i]:
                lines[i] = lines[i].split(':') 
                if lines[i][1] not in self.R_table:
                    self.R_table[lines[i][1]] = np.zeros(self.N_state)
                if lines[i][2] == ' * ':
                    self.R_table[lines[i][1]][int(lines[i][3])] = float(lines[i][5])
                else:
                    self.R_table[lines[i][1]][int(lines[i][2])] = float(lines[i][5])
                                     
            else:
                lines[i] = lines[i].split(' ')
                
            if lines[i][0] in ['#', 'agents:', 'discount:', 'values:', 'states:', 'start:', '0.0', '1.0']:
                continue
            
            elif lines[i][0] == 'actions:':
                self.actions = lines[i+1].split(' ')
                
            elif lines[i][0] == 'observations:':
                self.observations = lines[i+1].split(' ')

class DecTiger(Domain):
    
    def __init__(self, N_state, N_agent, N_ma, N_obs, Discount):
        print "Initiallize Problem Domain : DecTiger"
        Domain.__init__(self, N_state, N_agent, N_ma, N_obs, Discount)
        
        # define MAs
        self.MAs = list()
        self.MAs_names = ['Open_Left_Door', 'Open_Right_Door', 'Listen']
        self.MAs_rewards = [[-100, 10], [10, -100], [-1, -1]]
        for ma_id in range(N_ma):
            self.MAs.append(MA(ma_id, self.MAs_names[ma_id], self.MAs_rewards[ma_id]))

        # define states
        self.state_name = ['tiger-left', 'tiger-right']
        self.state_reset = [0.5, 0.5]
        
        # define observations
        self.observationprob = list()
        self.observation_name = ['hear-left', 'hear-right']
        
        # setup probability of observation given state and action
        self.observationprob = [[0.85, 0.15], [0.15, 0.85]]
        
        # setup horizon
        self.policyexecutedstep = 70 # how many step each agent can run 
            
    def evaljointpolicy(self):
        # set tiger state
        state_id = np.random.choice(range(self.N_state), 1, p=self.state_reset)[0]  
        #print "The initial state is ", state_id
        #values = list()
        value = 0.0
        for i in range(self.policyexecutedstep):
            rewards = self.MAs[self.agents[0].curMA].rewards[state_id] + self.MAs[self.agents[1].curMA].rewards[state_id]
            
            if rewards == -200:
                rewards = rewards / 4.0
            if rewards == -90:
                rewards = -100
            
            #print "Inmediately reward is ", rewards
                
            value += self.gamma**i * rewards
            #values.append(value)
             
            if (self.agents[0].curMA in [0, 1]) or (self.agents[1].curMA in [0, 1]):
                # sample the next state 
                state_id = np.random.choice(range(self.N_state), 1, p=self.state_reset)[0]
                #print "Next state is ", state_id 
              
            # sample observation and get next node for each agent
            if self.agents[0].curMA == 2 and self.agents[1].curMA == 2:
                cur_obs = np.random.choice(range(self.N_obs), 2, p=self.observationprob[state_id])
                #print "current obs is ", cur_obs
                for ag_id in range(self.N_agent):
                    self.agents[ag_id].setupNode(cur_obs[ag_id])           
            else:
                cur_obs = np.random.choice(range(self.N_obs), 2, p=[0.5, 0.5])
                #print "current obs is ", cur_obs
                for ag_id in range(self.N_agent):
                    self.agents[ag_id].setupNode(cur_obs[ag_id])
        
        # reset agents
        for ag_id in range(self.N_agent):
            self.agents[ag_id].resetag()    
                      
        return value 

class GridSmall(Domain): 
    
    def __init__(self, N_state, N_agent, N_ma, N_obs, Discount, horizon):
        print "Initiallize Problem Domain : GridSmall"
        path = 'problem_domains/GridSmall.dpomdp'
        Domain.__init__(self, N_state, N_agent, N_ma, N_obs, Discount)
        # load problem domain file
        self.dpomdp_parser(path)
        self.i_state = 6
        self.steps = horizon
        
    def evaljointpolicy(self):
        start = self.i_state
        value = 0.0
        for i in range(self.steps):
            # get joint action
            joint_action = ' ' + str(self.agents[0].curMA) + ' ' + str(self.agents[1].curMA) + ' '
            
            # sample next state
            next_state = np.random.choice(range(self.N_state), 1, p=self.T_table[joint_action][start,:])[0]
            
            # calculate rewards
            reward = self.R_table[' * '][next_state]
            value = value + self.gamma**i * reward
            
            # get observation and update controller node
            for ag_id in range(self.N_agent):
                self.agents[ag_id].setupNode(self.O_table[' * '][next_state][ag_id])
            
            start = next_state
                
        # reset agents
        for ag_id in range(self.N_agent):
            self.agents[ag_id].resetag()
        
        return value

class env(Domain):
    
    def __init__(self, domain_name, init_state, N_state, N_agent, N_ma, N_obs, Discount, horizon):
        print "Initiallize Problem Domain : " + domain_name
        path = './problem_domains/' + domain_name + '.dpomdp'
        Domain.__init__(self, N_state, N_agent, N_ma, N_obs, Discount)
        # load problem domain file
        self.dpomdp_parser(path)
        self.i_state = init_state
        self.steps = horizon
        self.domain = domain_name
        
    def evaljointpolicy(self):
        start = self.i_state
        value = 0.0
        for i in range(self.steps):
            # get joint action
            joint_action = ' ' + str(self.agents[0].curMA) + ' ' + str(self.agents[1].curMA) + ' '
            
            # sample next state
            next_state = np.random.choice(range(self.N_state), 1, p=self.T_table[joint_action][start,:])[0]
            
            # calculate rewards
            if self.domain == "gridsmall":
                reward = self.R_table[' * '][next_state]
            else:
                if joint_action not in self.R_table:
                    reward = 0.0
                else:
                    # calculate rewards
                    reward = self.R_table[joint_action][start]
            value = value + self.gamma**i * reward
                
            # get observation and update controller node
            if self.domain == "gridsmall":
                for ag_id in range(self.N_agent):
                    self.agents[ag_id].setupNode(self.O_table[' * '][next_state][ag_id])
            else:
                for ag_id in range(self.N_agent):
                    self.agents[ag_id].setupNode(self.O_table[joint_action][next_state][ag_id])
            
            start = next_state
      
        # reset agents
        for ag_id in range(self.N_agent):
            self.agents[ag_id].resetag()
        
        return value                
