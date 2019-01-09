from GDICE_Python.Plotting import *
from GDICE_Python.Scripts import extractResultsFromAllRuns
import os
import pickle
import numpy as np

def getParamIndices(uniqueVals, field, paramList):
    indices = [[] for _ in uniqueVals]
    for valIndex, val in enumerate(paramList):
        indices[np.where(getattr(paramList[valIndex], field) == uniqueVals)[0][0]].append(valIndex)

    indices = tuple(np.array(indexSet, dtype=int) for indexSet in indices)
    return indices

if __name__ == "__main__":
    plotDir = 'Plots'

    if os.path.isfile('compiledValues.npz') and os.path.isfile('compiledValueLabels.pkl'):
        values = np.load('compiledValues.npz')
        runResultsFound, iterValues, iterStdDev = tuple(values[item] for item in ['runResultsFound', 'iterValue', 'iterStdDev'])
        labels = pickle.load(open('compiledValueLabels.pkl','rb'))
        envNames, paramList = labels['envs'], labels['params']
    else:
        basePath = '/media/david/USB STICK/EndResCombined'
        runResultsFound, iterValues, iterStdDev, envNames, paramList, runDirs = extractResultsFromAllRuns(basePath, True)

    nVTIndices = np.array([i for i in range(len(paramList)) if paramList[i].valueThreshold is None], dtype=int)
    ntIterValues, ntIterStdDev, ntRunResultsFound = iterValues[:, :, nVTIndices, :], \
                                                    iterStdDev[:, :, nVTIndices, :], \
                                                    runResultsFound[:, :, nVTIndices]
    nVTParams = [paramList[i] for i in nVTIndices]
    # numEnvs*numParams*numIters
    meanVals, meanStdDev, runStdDev, runRange, numValues = \
        np.nanmean(ntIterValues, axis=0), np.nanmean(ntIterStdDev, axis=0), np.nanstd(ntIterValues, axis=0), \
        (np.nanmin(ntIterValues, axis=0), np.nanmax(ntIterValues, axis=0)), np.count_nonzero(ntRunResultsFound, axis=0)

    # Figure out what the values are for each parameter we're talking about
    numNodeValues = np.unique(np.array([p.numNodes for p in nVTParams], dtype=int))
    numNodeIndices = getParamIndices(numNodeValues, 'numNodes', nVTParams)
    numSampleValues = np.unique(np.array([p.numSamples for p in nVTParams], dtype=int))
    numSampleIndices = getParamIndices(numSampleValues, 'numSamples', nVTParams)
    numBestSampleValues = np.unique(np.array([p.numBestSamples for p in nVTParams], dtype=int))
    numBestSampleIndices = getParamIndices(numBestSampleValues, 'numBestSamples', nVTParams)
    for envIndex in range(meanVals.shape[0]):
        pass
        """
        What should I plot for each environment?
        Honestly, probably need to choose these params AND THEN do mean, std, range, etc calculations
        
        Effects of number of nodes on performance (for each num, plot performance averaged over all params with that num?)
        Effects of number of samples
        Effects of number of best samples
        Effects of a positive value threshold? Would kinda depend on environment. Honestly, probably best to throw these out for the other params
        """

    pass