from __future__ import absolute_import, print_function, division

import numpy as np
from time import time_ns
import matplotlib.pyplot as plt

from ..Policy import GraphPolicyController
from ..Domains.PackageDelivery.Domain import PackageDeliveryDomain as Domain

def crossEntropySearch():
    #plt.ion()
    numNodes = 13  # Number of nodes in GraphPolicyController
    alpha = 0.2  # Learning rate
    numTMAs = 13  # Number ot TMAs defined
    numObs = 13  # Number of observations in observation space
    N_k = 50  # Number of iterations
    N_s = 50  # Number of samples per iteration
    N_b = 5  # Number of best samples kept from each iteration
    bestTMAs = None
    bestTransitions = None

    """
    valueFigure = plt.figure()
    valueAxes = plt.axes(autoscale_on=True, xlabel='Iteration*sample', ylabel='Values')
    axisHandle = plt.plot(np.array([], dtype=np.int32), np.array([]), color='r', marker='+')[0]
    """

    mGraphPolicyController = GraphPolicyController(numNodes, alpha, numTMAs, numObs, N_s)
    bestValue = 0
    allValues = np.zeros((N_k*N_s))
    totalIterations = allValues.size

    for iteration in range(N_k):
        currentIterationValues = np.full(N_s, -100)  # Reset sample values
        mGraphPolicyController.sample(N_s)  # Sample N_s samples
        currentTime = time_ns()

        # Evaluate each sample
        for sampleIndex in range(N_s):
            print("Iteration ",(iteration*N_s+sampleIndex+1), " of ",totalIterations, ". Best value so far: ",bestValue)
            mGraphPolicyController.setGraph(sampleIndex)
            newValue = evalPolicy(mGraphPolicyController)[0]
            currentIterationValues[sampleIndex] = newValue
            if newValue > bestValue:  # Update best value
                bestValue = newValue
                bestTMAs, bestTransitions = mGraphPolicyController.getPolicyTable()

        allValues[iteration*N_s:iteration*N_s+N_s] = currentIterationValues  # Store a history of iterations

        # Plot values so far
        """
        plt.ioff()
        axisHandle.set_xdata(np.arange(1, (iteration+1)*N_s+1))
        axisHandle.set_ydata(allValues[:(iteration + 1) * N_s])
        plt.ion()
        """
    print('G-DICE finished in ', time_ns() - currentTime)  # Track runtime

    #Plot the full set of values and save
    plt.plot(allValues, color='b', marker='+')
    plt.savefig('crossEntropySearch.eps', dpi=600)






"""
Evaluate a given GraphPolicyController
Input:
  evalPolicy: GraphPolicyController object to evaluate, set at a sample index
Outputs:
  finalValue: Final average value of policy
  completionTime: nparray of completion time for each iteration (in timesteps in domain)
  successProb:
"""
def evalPolicy(policyController, numPackageGoal=9999, isOutputOn=False):
    maxRuns = 80
    values = np.zeros(maxRuns)
    completionTime = np.zeros(maxRuns)
    for runIndex in range(maxRuns):
        dom = Domain()
        dom.setPolicyForAllAgents(policyController)
        values[runIndex], completionTime[runIndex] = \
            dom.evaluateCurrentPolicy(numPackagesGoal=numPackageGoal, isOutputOn=isOutputOn)

    print("Average value ", np.mean(values))
    print("Best value ", np.max(values))

    valuePlotY = np.zeros(maxRuns)
    for valueIndex in range(maxRuns):
        valuePlotY[valueIndex] = np.sum(values[:valueIndex]) / (valueIndex+1)

    finalValue = valuePlotY[-1]
    successProb = np.sum(completionTime != -1) / completionTime.size  # Number of times completion time is valid
    completionTime = np.sum(completionTime[completionTime != 1]) / completionTime[completionTime != 1].size  # Sum of valid completion times
    return finalValue, successProb, completionTime






if __name__ == "__main__":
    crossEntropySearch()