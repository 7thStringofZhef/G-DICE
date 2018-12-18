import numpy as np
from gym_pomdps import list_pomdps
from .Parameters import GDICEParams
import os
import pickle
import glob


# Define a list of GDICE parameter objects that permute the variables across the possible values
def getGridSearchGDICEParams():
    N_n = np.arange(5, 16, 5)  # 5, 10, 15 nodes
    N_k = 1000  # Iterate, plot for every some # iterations
    N_s = np.arange(30, 71, 10)  # 30-70 samples per iteration (by 5)
    N_b = np.arange(3, 10, 2)  # Keep best 3, 5, 7, 9 samples
    N_sim = 1000  # 1000 simulations per sampled controller
    lr = np.array([0.05, 0.1, 0.2])  # Learning rate .05, .1, or .2
    vThresholds = [None, 0]  # Either no threshold or no non-negative values
    timeHorizon = 100  # Each simulation goes for 100 steps (or until episode ends)

    # All registered Pomdp environments, only the episodic versions, no rocksample
    envStrings = [pomdp for pomdp in list_pomdps() if 'episodic' not in pomdp and 'rock' not in pomdp]
    # Currently using just learning rate of 0.1
    paramList = [GDICEParams(n, N_k, j, N_sim, k, lr[1], t, timeHorizon) for n in N_n for j in N_s for k in N_b for l in lr for t in vThresholds]
    return envStrings, paramList

# Save the results of a run
def saveResults(baseDir, envName, testParams, results):
    print('Saving...')
    savePath = os.path.join(baseDir, 'GDICEResults', envName)  # relative to current path
    os.makedirs(savePath, exist_ok=True)
    bestValue, bestValueStdDev, bestActionTransitions, bestNodeObservationTransitions, updatedControllerDistribution, \
    estimatedConvergenceIteration, allValues, allStdDev, bestValueAtEachIteration, bestStdDevAtEachIteration = results
    np.savez(os.path.join(savePath, testParams.name)+'.npz', bestValue=bestValue, bestValueStdDev=bestValueStdDev,
             bestActionTransitions=bestActionTransitions, bestNodeObservationTransitions=bestNodeObservationTransitions,
             estimatedConvergenceIteration=estimatedConvergenceIteration, allValues=allValues, allStdDev=allStdDev,
             bestValueAtEachIteration=bestValueAtEachIteration, bestStdDevAtEachIteration=bestStdDevAtEachIteration)
    pickle.dump(updatedControllerDistribution, open(os.path.join(savePath, testParams.name)+'.pkl', 'wb'))
    pickle.dump(testParams, open(os.path.join(savePath, testParams.name+'_params') + '.pkl', 'wb'))


# Load the results of a run
# Inputs:
#   filePath: Path to any of the files.
# Outputs:
#   All the results generated
#   The updated controller distribution
#   The GDICE params object
def loadResults(filePath):
    baseName = os.path.splitext(filePath)[0]
    print('Loading...')
    fileDict = np.load(baseName+'.npz')
    keys = ('bestValue', 'bestValueStdDev', 'bestActionTransitions', 'bestNodeObservationTransitions',
            'estimatedConvergenceIteration', 'allValues', 'allStdDev', 'bestValueAtEachIteration',
            'bestStdDevAtEachIteration')
    results = tuple([fileDict[key] for key in keys])
    updatedControllerDistribution = pickle.load(open(baseName+'.pkl', 'rb'))
    params = pickle.load(open(baseName + '_params.pkl', 'rb'))
    return results, updatedControllerDistribution, params

# Check if a particular permutation is finished
#   It's finished if its files can be found in the end results directory (instead of the temp)
#   Returns whether it's finished as well as the filename
def checkIfFinished(envStr, paramsName, endDir='EndResults', baseDir=''):
    npzName = os.path.join(baseDir, endDir, envStr, paramsName+'.npz')
    return os.path.isfile(npzName), npzName

# Check if a particular permutation is partially run
#   It's partially run if its files can be found in the end results directory (instead of the temp)
#   Returns whether it's started as well as the filename
def checkIfPartial(envStr, paramsName, tempDir='GDICEResults', baseDir=''):
    npzName = os.path.join(baseDir, tempDir, envStr, paramsName+'.npz')
    return os.path.isfile(npzName), npzName

# Attempt to delete all temp results for runs that are finished
def deleteFinishedTempResults(basedirs=np.arange(1,11,dtype=int)):
    envList, GDICEList = getGridSearchGDICEParams()
    for rundir in basedirs:
        for envStr in envList:
            for params in GDICEList:
                if checkIfFinished(envStr, params.name, baseDir=rundir):  # if run is finished
                    # Delete the temp results
                    try:
                        for filename in glob.glob(os.path.join(rundir, 'GDICEResults', params.name) + '*'):
                            os.remove(filename)
                    except:
                        continue


