import numpy as np
from gym_pomdps import list_pomdps
from gym_dpomdps import list_dpomdps
from GDICE_Python.Parameters import GDICEParams
import os
import pickle
import glob
import traceback
import filelock


# Define a list of GDICE parameter objects that permute the variables across the possible values
def getGridSearchGDICEParams():
    N_n = np.arange(5, 16, 5)  # 5, 10, 15 nodes
    N_k = 1000  # Iterate, plot for every some # iterations
    N_s = np.arange(30, 71, 20)  # 30-70 samples per iteration (by 20)
    N_b = np.arange(3, 8, 2)  # Keep best 3, 5, 7 samples
    N_sim = 500  # 500 simulations per sampled controller
    lr = [0.02, 0.1, 0.5]  # Learning rate .02, .1, or .5
    vThresholds = [None, 0]  # Either no threshold or no non-negative values. UNUSED
    timeHorizon = 100  # Each simulation goes for 100 steps (or until episode ends)

    # All registered Pomdp environments, only the non-episodic versions, no rocksample
    envStrings = ["POMDP-1d-v0", "POMDP-4x3-v0", "POMDP-cheese-v0", "POMDP-concert-v0", "POMDP-hallway-v0", "POMDP-heavenhell-v0", "POMDP-loadunload-v0", "POMDP-network-v0", "POMDP-tiger-v0", "POMDP-voicemail-v0"]
    #envStrings = [pomdp for pomdp in list_pomdps() if 'episodic' not in pomdp and 'rock' not in pomdp]
    paramList = [GDICEParams(n, N_k, j, N_sim, k, l, None, timeHorizon) for n in N_n for j in N_s for l in lr for k in N_b]
    return envStrings, paramList

# Define a list of GDICE parameter objects that permute the variables across the possible values
def getGridSearchGDICEParamsDPOMDP():
    N_n = np.arange(5, 16, 5)  # 5, 10, 15 nodes. Currently, each dpomdp has 2 agents, and each agent has same # nodes
    centralized = [False, True]  # One distribution for all agents, or 1 for each?
    N_k = 1000  # Iterate, plot for every some # iterations
    N_s = np.arange(30, 71, 20)  # 30-70 samples per iteration (by 20)
    N_b = np.arange(3, 8, 2)  # Keep best 3, 5, 7 samples
    N_sim = 1000  # 1000 simulations per sampled controller
    lr = np.array([0.02, 0.1, 0.5])  # Learning rate .05, .1, or .2
    vThresholds = [None, 0]  # Either no threshold or no non-negative values. UNUSED
    timeHorizon = 100  # Each simulation goes for 100 steps (or until episode ends)

    # All registered dpomdp environments, only the non-episodic versions
    envStrings = [dpomdp for dpomdp in list_dpomdps() if 'episodic' not in dpomdp]
    paramList = [GDICEParams(n, N_k, j, N_sim, k, l, None, timeHorizon, c) for n in N_n for c in centralized for j in N_s for k in N_b for l in lr]
    return envStrings, paramList

# Write out grid search permutation to text
def writePOMDPGridSearchParamsToFile(filepath='POMDPsToEval.txt', numRuns=5):
    envStrings, paramList = getGridSearchGDICEParams()
    filelist = [os.path.join(str(run), env, param.name) for run in np.arange(numRuns)+1 for env in envStrings for param in paramList]
    with open(filepath, 'w') as f:
        for fileStr in filelist:
            f.write(fileStr+'\n')

def writeDPOMDPGridSearchParamsToFile(filepath='DPOMDPsToEval.txt', numRuns=5):
    envStrings, paramList = getGridSearchGDICEParamsDPOMDP()
    filelist = [os.path.join(str(run), env, param.name) for run in np.arange(numRuns)+1 for env in envStrings for param in paramList]
    with open(filepath, 'w') as f:
        for fileStr in filelist:
            f.write(fileStr+'\n')

# Claim the next param set
def claimRunEnvParamSet(filepath='POMDPsToEval.txt'):
    # Lock file
    with filelock.FileLock(filepath+'.lock'):
        with open(filepath, 'r+') as f:
            lines = f.readlines()
            if not lines:  # No more environments
                return None
            nextSet = lines[0]  # Claim the next set
            f.seek(0)
            if len(lines) > 1:  # Write all but the next set
                for line in lines[1:]:
                    f.write(line)
            else:
                f.write('')
            f.truncate()

    # Move to "in progress"
    # Use current process id to signal who is working on it
    inProgFilepath = os.path.splitext(filepath)[0]+'_inprog.txt'
    with filelock.FileLock(inProgFilepath+'.lock'):
        with open(inProgFilepath, 'a') as f:
            f.write(nextSet)
            #f.write(str(os.getpid())+' '+nextSet)
    return nextSet.rstrip()

# Claim the next param set from the "inprogress" file, assuming no one's working on it
def claimRunEnvParamSet_unfinished(filepath='POMDPsToEval.txt'):
    inProgFilepath = os.path.splitext(filepath)[0] + '_inprog.txt'
    # Lock file
    with filelock.FileLock(inProgFilepath+'.lock'):
        with open(inProgFilepath, 'r+') as f:
            lines = f.readlines()
            if not lines:  # No more environments
                return None
            nextSet = lines[0]  # Claim the next set
            f.seek(0)
            if len(lines) > 1:  # Write all but the next set
                for line in lines[1:]:
                    f.write(line)
            else:
                f.write('')
            f.truncate()

    return nextSet.rstrip()

def registerRunEnvParamSetCompletion(doneSet, filepath='POMDPsToEval.txt'):
    inProgFilepath = os.path.splitext(filepath)[0] + '_inprog.txt'
    completionPath = os.path.splitext(filepath)[0]+'_done.txt'
    #pid = str(os.getpid())
    # Update completion file
    with filelock.FileLock(completionPath+'.lock'):
        with open(completionPath, 'a') as f:
            f.write(doneSet+'\n')

    # Update in progress file (find and remove line)
    with filelock.FileLock(inProgFilepath+'.lock'):
        with open(inProgFilepath, 'r+') as f:
            lines = f.readlines()
            try:
                #lines.remove(pid + ' ' + doneSet + '\n')
                lines.remove(doneSet+'\n')
            except:
                pass
            f.seek(0)
            for line in lines:
                f.write(line)
            f.truncate()

def registerRunEnvParamSetCompletion_unfinished(doneSet, filepath='POMDPsToEval.txt'):
    completionPath = os.path.splitext(filepath)[0]+'_done.txt'
    #pid = str(os.getpid())
    # Update completion file
    with filelock.FileLock(completionPath+'.lock'):
        with open(completionPath, 'a') as f:
            f.write(doneSet+'\n')


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
    npzName = os.path.join(baseDir, endDir, 'GDICEResults', envStr, paramsName+'.npz')
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
                if checkIfFinished(envStr, params.name, baseDir=str(rundir))[0]:  # if run is finished
                    # Delete the temp results
                    try:
                        for filename in glob.glob(os.path.join(str(rundir), 'GDICEResults', params.name) + '*'):
                            os.remove(filename)
                    except:
                        continue

def replaceResultsWithDummyFiles(baseDirs = np.arange(1,11,dtype=int), endDir='EndResults'):
    for dir in baseDirs:
        startPath = os.path.join(str(dir), endDir, 'GDICEResults')
        for envDir in os.listdir(startPath):
            envPath = os.path.join(startPath, envDir)
            for file in os.listdir(envPath):
                open(file, 'w').close()  # replace with blank file of same name

# Script to get me the results from all the runs beneath a directory
# Structure is basePath/<1,2,3,4,5,6,7,8,9,10>/EndResults/GDICEResults/envName/<params>
def extractResultsFromAllRuns(basePath, saveSeparate=False, saveDummyList=True):
    # List environments I actually did runs for
    baseEnvNames = ['1d', '4x3', 'cheese', 'concert', 'hallway', 'heavenhell', 'loadunload', 'network', 'shopping_5', 'tiger', 'voicemail']
    envNames = ['POMDP-' + base + '-v0' for base in baseEnvNames]
    runDirs = np.arange(1, 11, dtype=int)
    f = None

    if saveDummyList:
        f = open("FilesToGenList.txt", 'w')
        f2 = open("UnfinishedRuns.txt", 'w')


    # Get list of grid search params
    paramList = getGridSearchGDICEParams()[1]

    # Keep track of which run results we actually found
    runResultsFound = np.zeros((len(runDirs), len(envNames), len(paramList)), dtype=bool)

    # I just want the best value (and its stddev) at each iteration
    iterValues = np.full((len(runDirs), len(envNames), len(paramList), paramList[0].numIterations), np.nan, dtype=np.float64)
    iterStdDev = np.full((len(runDirs), len(envNames), len(paramList), paramList[0].numIterations), np.nan, dtype=np.float64)

    # Start digging!
    for runDir in runDirs:
        runPath = os.path.join(basePath, str(runDir), 'EndResults', 'GDICEResults')
        for envIndex in range(len(envNames)):
            envName = envNames[envIndex]
            envPath = os.path.join(runPath, envName)
            for paramIndex in range(len(paramList)):
                param = paramList[paramIndex]
                filePath = os.path.join(envPath, param.name+'.npz')
                if os.path.isfile(filePath):
                    try:
                        fileResults = np.load(filePath)
                        iterValues[runDir-1, envIndex, paramIndex, :] = fileResults['bestValueAtEachIteration']
                        iterStdDev[runDir-1, envIndex, paramIndex, :] = fileResults['bestStdDevAtEachIteration']
                        runResultsFound[runDir-1, envIndex, paramIndex] = True
                        fileResults.close()
                        if saveDummyList:
                            strippedPath = filePath[filePath.find('/'+str(runDir))+1:]
                            f.write(strippedPath + '\n')
                    except:
                        print('Load failed for env ' + envName + ' params ' + os.path.basename(filePath))
                        continue
                else:
                    strippedPath = filePath[filePath.find('/'+str(runDir))+1:]
                    f2.write(strippedPath+'\n')
    # Save the extracted results in a single file for easier access later
    if saveSeparate:
        try:
            np.savez('compiledValues.npz', runResultsFound=runResultsFound, iterValue=iterValues, iterStdDev=iterStdDev, runs=runDirs)
            pickle.dump({'envs': envNames, 'params': paramList}, open('compiledValueLabels.pkl', 'wb'))
        except Exception as e:
            traceback.print_exc()
            print(e)
            print('Save failed')

    return runResultsFound, iterValues, iterStdDev, envNames, paramList, runDirs

# Generate dummy files from a text list
def genDummyFilesFromList(pathToList, baseGenPath='/scratch/slayback.d/GDICE'):
    pathToList = 'FilesToGenList.txt'
    baseGenPath = ''
    listF = open(pathToList, 'r')
    for f in listF.readlines()[:50]:
        filepath = os.path.join(baseGenPath, f)
        try:
            if not (os.path.exists(os.path.dirname(filepath))):
                os.makedirs(os.path.dirname(filepath))
            open(filepath, 'w').close()
        except:
            continue


if __name__ == "__main__":
    writePOMDPGridSearchParamsToFile(numRuns=2)
    writeDPOMDPGridSearchParamsToFile(numRuns=2)

