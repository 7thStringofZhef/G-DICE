from GDICE_Python.Plotting import *
from GDICE_Python.Scripts import extractResultsFromAllRuns
import os
import pickle
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

def getParamIndices(uniqueVals, field, paramList):
    indices = [[] for _ in uniqueVals]
    for valIndex, val in enumerate(paramList):
        indices[np.where(getattr(paramList[valIndex], field) == uniqueVals)[0][0]].append(valIndex)

    indices = tuple(np.array(indexSet, dtype=int) for indexSet in indices)
    return indices


def extractNonEntropy():
    plotDir = 'Plots'

    if os.path.isfile('compiledValues.npz') and os.path.isfile('compiledValueLabels.pkl'):
        values = np.load('compiledValues.npz')
        runResultsFound, iterValues, iterStdDev = tuple(values[item] for item in ['runResultsFound', 'iterValue', 'iterStdDev'])
        labels = pickle.load(open('compiledValueLabels.pkl','rb'))
        envNames, paramList = labels['envs'], labels['params']
    else:
        basePath = '/home/david/GDiceRes'
        runResultsFound, iterValues, iterStdDev, envNames, paramList, runDirs = extractResultsFromAllRuns(basePath, True)

    # Scrap value threshold for now
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
    iterations = np.arange(meanVals.shape[-1])
    os.makedirs('Figures', exist_ok=True)
    for envIndex in range(meanVals.shape[0]):
        # num nodes
        nodeMeanVals = [np.nanmean(meanVals[envIndex, nodeValIndex, :], axis=0) for nodeValIndex in numNodeIndices]  # 1000
        plt.plot(iterations, nodeMeanVals[0], label=str(numNodeValues[0]))
        plt.plot(iterations, nodeMeanVals[1], label=str(numNodeValues[1]))
        plt.plot(iterations, nodeMeanVals[2], label=str(numNodeValues[2]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.title(envNames[envIndex] + ' number of nodes average performance')
        plt.legend()
        plt.savefig('Figures/'+envNames[envIndex]+'Nodes.png')
        plt.cla()

        # num samples per iteration
        sampleMeanVals = [np.nanmean(meanVals[envIndex, numSampleIndex, :], axis=0) for numSampleIndex in numSampleIndices]  # 1000
        plt.plot(iterations, sampleMeanVals[0], label=str(numSampleValues[0]))
        plt.plot(iterations, sampleMeanVals[1], label=str(numSampleValues[1]))
        plt.plot(iterations, sampleMeanVals[2], label=str(numSampleValues[2]))
        plt.plot(iterations, sampleMeanVals[3], label=str(numSampleValues[3]))
        plt.plot(iterations, sampleMeanVals[4], label=str(numSampleValues[4]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.title(envNames[envIndex] + ' number of samples average performance')
        plt.legend()
        plt.savefig('Figures/'+envNames[envIndex]+'Samples.png')
        plt.cla()

        # num best samples per iteration
        bestSampleMeanVals = [np.nanmean(meanVals[envIndex, numBSampleIndex, :], axis=0) for numBSampleIndex in numBestSampleIndices]  # 1000
        plt.plot(iterations, bestSampleMeanVals[0], label=str(numBestSampleValues[0]))
        plt.plot(iterations, bestSampleMeanVals[1], label=str(numBestSampleValues[1]))
        plt.plot(iterations, bestSampleMeanVals[2], label=str(numBestSampleValues[2]))
        plt.plot(iterations, bestSampleMeanVals[3], label=str(numBestSampleValues[3]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.title(envNames[envIndex] + ' number of best samples average performance')
        plt.legend()
        plt.savefig('Figures/'+envNames[envIndex]+'BestSamples.png')
        plt.cla()

def extractEntropy():
    plotDir = 'Plots'

    if os.path.isfile('compiledValuesEnt.npz') and os.path.isfile('compiledValueLabelsEnt.pkl'):
        values = np.load('compiledValuesEnt.npz')
        runResultsFound, iterValues, iterStdDev = tuple(values[item] for item in ['runResultsFound', 'iterValue', 'iterStdDev'])
        labels = pickle.load(open('compiledValueLabelsEnt.pkl','rb'))
        envNames, paramList = labels['envs'], labels['params']
    else:
        basePath = '/home/david/GDiceRes/Entropy'
        runResultsFound, iterValues, iterStdDev, envNames, paramList, runDirs = extractResultsFromAllRuns(basePath, True)

    # Scrap value threshold for now
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
    ceDeceIndices = getParamIndices([False, True], 'centralized', nVTParams)
    iterations = np.arange(meanVals.shape[-1])
    os.makedirs('Figures', exist_ok=True)
    for envIndex in range(meanVals.shape[0]):
        # num nodes
        nodeMeanVals = [np.nanmean(meanVals[envIndex, nodeValIndex, :], axis=0) for nodeValIndex in numNodeIndices]  # 1000
        plt.plot(iterations, nodeMeanVals[0], label=str(numNodeValues[0]))
        plt.plot(iterations, nodeMeanVals[1], label=str(numNodeValues[1]))
        plt.plot(iterations, nodeMeanVals[2], label=str(numNodeValues[2]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.title(envNames[envIndex] + ' number of nodes average performance')
        plt.legend()
        plt.savefig('Figures/'+envNames[envIndex]+'Nodes.png')
        plt.cla()

        # num samples per iteration
        sampleMeanVals = [np.nanmean(meanVals[envIndex, numSampleIndex, :], axis=0) for numSampleIndex in numSampleIndices]  # 1000
        plt.plot(iterations, sampleMeanVals[0], label=str(numSampleValues[0]))
        plt.plot(iterations, sampleMeanVals[1], label=str(numSampleValues[1]))
        plt.plot(iterations, sampleMeanVals[2], label=str(numSampleValues[2]))
        plt.plot(iterations, sampleMeanVals[3], label=str(numSampleValues[3]))
        plt.plot(iterations, sampleMeanVals[4], label=str(numSampleValues[4]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.title(envNames[envIndex] + ' number of samples average performance')
        plt.legend()
        plt.savefig('Figures/'+envNames[envIndex]+'Samples.png')
        plt.cla()

        # num best samples per iteration
        bestSampleMeanVals = [np.nanmean(meanVals[envIndex, numBSampleIndex, :], axis=0) for numBSampleIndex in numBestSampleIndices]  # 1000
        plt.plot(iterations, bestSampleMeanVals[0], label=str(numBestSampleValues[0]))
        plt.plot(iterations, bestSampleMeanVals[1], label=str(numBestSampleValues[1]))
        plt.plot(iterations, bestSampleMeanVals[2], label=str(numBestSampleValues[2]))
        plt.plot(iterations, bestSampleMeanVals[3], label=str(numBestSampleValues[3]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.title(envNames[envIndex] + ' number of best samples average performance')
        plt.legend()
        plt.savefig('Figures/'+envNames[envIndex]+'BestSamples.png')
        plt.cla()

if __name__ == "__main__":
    plotDir = 'Plots'

    if os.path.isfile('compiledValues.npz') and os.path.isfile('compiledValueLabels.pkl'):
        values = np.load('compiledValues.npz')
        runResultsFound, iterValues, iterStdDev = tuple(values[item] for item in ['runResultsFound', 'iterValue', 'iterStdDev'])
        labels = pickle.load(open('compiledValueLabels.pkl','rb'))
        envNames, paramList = labels['envs'], labels['params']
    else:
        basePath = '/home/david/GDiceRes'
        runResultsFound, iterValues, iterStdDev, envNames, paramList, runDirs = extractResultsFromAllRuns(basePath, True)

    # Scrap value threshold for now
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
    iterations = np.arange(meanVals.shape[-1])
    os.makedirs('Figures', exist_ok=True)
    for envIndex in range(meanVals.shape[0]):
        # num nodes
        nodeMeanVals = [np.nanmean(meanVals[envIndex, nodeValIndex, :], axis=0) for nodeValIndex in numNodeIndices]  # 1000
        plt.plot(iterations, nodeMeanVals[0], label=str(numNodeValues[0]))
        plt.plot(iterations, nodeMeanVals[1], label=str(numNodeValues[1]))
        plt.plot(iterations, nodeMeanVals[2], label=str(numNodeValues[2]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.title(envNames[envIndex] + ' number of nodes average performance')
        plt.legend()
        plt.savefig('Figures/'+envNames[envIndex]+'Nodes.png')
        plt.cla()

        # num samples per iteration
        sampleMeanVals = [np.nanmean(meanVals[envIndex, numSampleIndex, :], axis=0) for numSampleIndex in numSampleIndices]  # 1000
        plt.plot(iterations, sampleMeanVals[0], label=str(numSampleValues[0]))
        plt.plot(iterations, sampleMeanVals[1], label=str(numSampleValues[1]))
        plt.plot(iterations, sampleMeanVals[2], label=str(numSampleValues[2]))
        plt.plot(iterations, sampleMeanVals[3], label=str(numSampleValues[3]))
        plt.plot(iterations, sampleMeanVals[4], label=str(numSampleValues[4]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.title(envNames[envIndex] + ' number of samples average performance')
        plt.legend()
        plt.savefig('Figures/'+envNames[envIndex]+'Samples.png')
        plt.cla()

        # num best samples per iteration
        bestSampleMeanVals = [np.nanmean(meanVals[envIndex, numBSampleIndex, :], axis=0) for numBSampleIndex in numBestSampleIndices]  # 1000
        plt.plot(iterations, bestSampleMeanVals[0], label=str(numBestSampleValues[0]))
        plt.plot(iterations, bestSampleMeanVals[1], label=str(numBestSampleValues[1]))
        plt.plot(iterations, bestSampleMeanVals[2], label=str(numBestSampleValues[2]))
        plt.plot(iterations, bestSampleMeanVals[3], label=str(numBestSampleValues[3]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.title(envNames[envIndex] + ' number of best samples average performance')
        plt.legend()
        plt.savefig('Figures/'+envNames[envIndex]+'BestSamples.png')
        plt.cla()
        """
        What should I plot for each environment?
        Honestly, probably need to choose these params AND THEN do mean, std, range, etc calculations
        
        Effects of number of nodes on performance (for each num, plot performance averaged over all params with that num?)
        Effects of number of samples
        Effects of number of best samples
        Effects of a positive value threshold? Would kinda depend on environment. Honestly, probably best to throw these out for the other params
        """

    pass