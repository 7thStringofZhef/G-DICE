#!/usr/bin/env python

import numpy as np
import multiprocessing
import math
import argparse

from Policy_Controller import GraphPolicyController
from Domain import DecTiger,env

parser = argparse.ArgumentParser(description='Domain Name and GDICE parameters')

# domain param
parser.add_argument('--domain', default="", help="problem domain")
parser.add_argument('--n_agent', type=int, default=2, help="number of agents")
parser.add_argument('--n_state', type=int, default=0, help="state space")
parser.add_argument('--n_action', type=int, default=0, help="action space")
parser.add_argument('--n_obs', type=int, default=0, help="observation space")
parser.add_argument('--init_state', type=int, default=0, help="initial state")
parser.add_argument('--horizon', type=int, default=0, help="horizon of the domain")
# GDICE param
parser.add_argument('--n_node', type=int, default=10, help="number of nodes in each FSA controller")
parser.add_argument('--n_iter', type=int, default=100, help="number of training iterations")
parser.add_argument('--n_sample', type=int, default=100, help="number of sampled policies in each iteration")
parser.add_argument('--n_best', type=int, default=10, help="number of the good samples picked out for updating")
# hyper param
parser.add_argument('--alpha', type=float, default=0.2, help="learning rate")
parser.add_argument('--discount', type=float, default=1.0, help="discount number")
parser.add_argument('--eval_time', type=int, default=1000, help="evaluation times per sampled policy")
# parallel processing control
parser.add_argument('--n_cpu', type=int, default=8, help="number of CPU cores used for parallel processing")

# evaluate the current sample policy and return the average performance value
def evaluate_one_sample(PolicyControllers, evaluation_times, s_id, p_domain):
    values_per_time = list()

    # set up joint-policy for each robot
    for Rob_id in range(params.n_agent):
        PolicyControllers[Rob_id].setupPolicy(s_id)
    
    p_domain.inputPolicyToAgents(PolicyControllers)
        
    for t in range(evaluation_times):      
        values_per_time.append(p_domain.evaljointpolicy())
        
    # calculate the average value of per sample joint-policy
    meanV = np.mean(values_per_time)
    s_error = np.std(values_per_time) / np.sqrt(len(values_per_time))
    return meanV, s_error

def multi_process_evaluation(num_sample, nprocs, controllers, problem_domain, times):
    def worker(num_sample, out_q1, out_q2, game):
        out_dict1 = {}
        out_dict2 = {}
        for s_id in num_sample:
            out_dict1[s_id], out_dict2[s_id] = evaluate_one_sample(controllers, times, s_id, game)

        out_q1.put(out_dict1)
        out_q2.put(out_dict2)
        
    out_q1 = multiprocessing.Queue()
    out_q2 = multiprocessing.Queue()  
    chunksize = int(math.ceil(len(num_sample) / float(nprocs)))
    procs = []
    for i in range(nprocs):
        p = multiprocessing.Process(
            target=worker,
            args=(num_sample[chunksize * i:chunksize * (i + 1)],
                  out_q1, out_q2, problem_domain))
        procs.append(p)
        p.start()   
    
    resultdict1 = {}
    resultdict2 = {}
    for i in range(nprocs):
        resultdict1.update(out_q1.get())
        resultdict2.update(out_q2.get())
    
    for p in procs:
        p.join()   
        
    return resultdict1.values(), resultdict2.values()
        
if __name__ == '__main__':

    params = parser.parse_args()
    
    # initialize policy graph with N_n nodes and N_OB edges per node for each robot
    FSAs = list()
    for i in range(params.n_agent):
        FSA = GraphPolicyController(params.n_action, params.n_obs, params.n_sample, params.n_node, params.alpha) 
        FSAs.append(FSA)
    
    # initiallize best and worst value
    V_b = -float('inf')
    V_w = -float('inf')
    
    # the standard error of current best policy
    V_bstd_error = float('inf')
    
    # collect all values during iterations
    all_values = list()
    
    # initiallize best joint-policy
    best_policy = list()
    
    # setup problem domain    
    if params.domain == "dectiger":
        scenario = DecTiger(2, params.n_agent, params.n_action, params.n_obs, params.discount)
    else:
        scenario = env(params.domain, params.init_state, params.n_state, params.n_agent, params.n_action, params.n_obs, params.discount, params.horizon)
        
    # start iteration    
    for k in range(params.n_iter):
        # initilize the best N_b policies' list
        N_b_policiesID = list()
        N_b_policiesV = list()
        
        # creat N_s sample joint-policies for each robot
        for Rob_id in range(params.n_agent):
            FSAs[Rob_id].sample()
                   
        # evaluate sample policies
        nums = range(params.n_sample)
        values, standard_errors = multi_process_evaluation(nums, params.n_cpu, FSAs, scenario, params.eval_time)
                          
        # collecting samples' values
        all_values += values
        
        sorted_idx = sorted(range(len(values)), key=lambda x:values[x])
        sorted_idx.reverse()
        sorted_values = values
        sorted_values.sort(reverse=True)
        
        
        # collect the best N_b samples and update V_b and V_w
        for s_id in range(params.n_sample):
            if len(N_b_policiesID) == params.n_best:
                # already collect the best N_b samples or no more value bigger than V_w
                V_w = sorted_values[s_id - 1]
                break
            if sorted_values[s_id] > V_b:
                V_bstd_error = standard_errors[sorted_idx[s_id]]
                V_b = sorted_values[s_id]
                # delete the old best policy
                del best_policy[:]
                # update the best joint-policy for all robots
                for i in range(params.n_agent):
                    best_policy.append([FSAs[i].getPolicy(sorted_idx[s_id])])
                
            if sorted_values[s_id] >= V_w:
                N_b_policiesV.append(sorted_values[s_id])
                N_b_policiesID.append(sorted_idx[s_id])
            else:
                break
            
        if len(N_b_policiesV) != 0:
            V_w = min(N_b_policiesV)
                  
        # update sampling distributions' parameters for each node of each robot's FSA
        for Rob_id in range(params.n_agent):
            FSAs[Rob_id].update_sample_distr(N_b_policiesID)
        
        
        print len(N_b_policiesV), " best policies were picked out."    
        print "The current best policy's value(standard error) after ", k, "th iteration is ", V_b, "(", V_bstd_error, ")"
        print "The current best policy: "
        for i in range(params.n_agent):
            print "Agent", i+1, "'s controller's actions"
            print best_policy[i][0][0]
            print "Agent", i+1, "'s controller's transitions"
            print best_policy[i][0][1]
    
    

    
                        
            
