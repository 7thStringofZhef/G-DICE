from __future__ import absolute_import, print_function, division

import numpy as np
from time import time
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

from GDICEPythonOld.Policy import GraphPolicyController
from GDICEPythonOld.Domains.PackageDelivery.Domain import PackageDeliveryDomain as Domain

"""
Run G-Dice on the PackageDelivery domain with the given parameters

Inputs:
  numNodes: Number of nodes in GraphPolicyController
  alpha: Learning rate
  numTMAs: Number ot TMAs defined
  numObs: Number of observations in observation space
  N_k: Number of iterations
  N_s: Number of samples per iteration
  N_b: Number of best samples kept from each iteration
Outputs:
  bestValue: Best policy value seen
  bestTMAs, bestTransitions: Returns from policycontroller that gave best value
"""
def crossEntropySearch(numNodes=13, alpha=0.2, numTMAs=13, numObs=13, N_k=50, N_s=50, N_b=5):
    #plt.ion()
    bestTMAs = []
    bestTransitions = []

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
        currentTime = time()

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
        mGraphPolicyController.updateProbs(currentIterationValues, N_b)

        # Plot values so far
        """
        plt.ioff()
        axisHandle.set_xdata(np.arange(1, (iteration+1)*N_s+1))
        axisHandle.set_ydata(allValues[:(iteration + 1) * N_s])
        plt.ion()
        """
        print('G-DICE iteration ',iteration,' finished in ', time() - currentTime)  # Track runtime

    #Plot the full set of values and save
    plt.plot(allValues, color='b', marker='+')
    plt.savefig('crossEntropySearch.eps', dpi=600)
    return bestValue, bestTMAs, bestTransitions

# Function to call for parallel samples (iterations can't be parallel). Returns evaluated value
def _crossEntropySearchOneSample(policyController, sampleIndex):
    policyController.setGraph(sampleIndex)
    return evalPolicy(policyController)[0]

"""
Run G-Dice on the PackageDelivery domain with the given parameters. Parallelizes across samples with max #workers

Inputs:
  numNodes: Number of nodes in GraphPolicyController
  alpha: Learning rate
  numTMAs: Number ot TMAs defined
  numObs: Number of observations in observation space
  N_k: Number of iterations
  N_s: Number of samples per iteration
  N_b: Number of best samples kept from each iteration
Outputs:
  bestValue: Best policy value seen
  bestTMAs, bestTransitions: Returns from policycontroller that gave best value
"""
def crossEntropySearchParallel(numNodes=10, alpha=0.2, numTMAs=13, numObs=13, N_k=300, N_s=30, N_b=3):
    # Set up pool with max # workers
    pool = multiprocessing.Pool()

    bestTMAs = None
    bestTransitions = None

    mGraphPolicyController = GraphPolicyController(numNodes, alpha, numTMAs, numObs, N_s)

    bestValue = 0
    allValues = np.zeros((N_k*N_s))
    totalIterations = allValues.size
    for iteration in range(N_k):
        print("Iteration ", (iteration * N_s + 1), " of ", totalIterations, ". Best value so far: ", bestValue)
        mGraphPolicyController.sample(N_s)  # Sample N_s samples

        sampleFunc = partial(_crossEntropySearchOneSample, mGraphPolicyController)  # Define a partial functions for this iteration
        poolResults = pool.map(sampleFunc, range(N_s))
        currentIterationValues = np.array(poolResults)

        # Update best value
        for sampleIndex in range(N_s):
            if currentIterationValues[sampleIndex] > bestValue:
                bestValue = currentIterationValues[sampleIndex]
                mGraphPolicyController.setGraph(sampleIndex)
                bestTMAs, bestTransitions = mGraphPolicyController.getPolicyTable()

        allValues[iteration*N_s:iteration*N_s+N_s] = currentIterationValues  # Store a history of iterations
        mGraphPolicyController.updateProbs(currentIterationValues, N_b)

    #Plot the full set of values and save
    pool.close()
    pool.join()
    plt.plot(allValues, color='b', marker='+')
    plt.savefig('crossEntropySearch.eps', dpi=600)
    return bestValue, bestTMAs, bestTransitions



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