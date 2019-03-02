from GDICE_Python.Scripts import extractResultsFromAllRuns, extractResultsFromAllRunsDPOMDP
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


def extractPOMDP():
    if os.path.isfile('compiledValuesPOMDP.npz') and os.path.isfile('compiledValueLabelsPOMDP.pkl'):
        values = np.load('compiledValuesPOMDP.npz')
        runResultsFound, iterValues, iterStdDev = tuple(values[item] for item in ['runResultsFound', 'iterValue', 'iterStdDev'])
        labels = pickle.load(open('compiledValueLabelsPOMDP.pkl', 'rb'))
        envNames, paramList = labels['envs'], labels['params']
        values = np.load('compiledValuesPOMDPEnt.npz')
        runResultsFoundE, iterValuesE, iterStdDevE = tuple(values[item] for item in ['runResultsFound', 'iterValue', 'iterStdDev'])
        labelsE = pickle.load(open('compiledValueLabelsPOMDPEnt.pkl', 'rb'))
        envNamesE, paramListE = labelsE['envs'], labelsE['params']
    else:
        basePath = '/home/david/GDiceRes'
        entPath = '/home/david/GDiceRes/Entropy'
        #runs*envs*params
        #runs*envs*params*iters
        #runs*envs*params*iters
        runResultsFound, iterValues, iterStdDev, envNames, paramList, runDirs = extractResultsFromAllRuns(basePath, True, sepName='compiledValuesPOMDP.npz', sepName2='compiledValueLabelsPOMDP.pkl')
        runResultsFoundE, iterValuesE, iterStdDevE, envNamesE, paramListE, runDirsE = extractResultsFromAllRuns(entPath, True, sepName='compiledValuesPOMDPEnt.npz', sepName2='compiledValueLabelsPOMDPEnt.pkl')

    # Scrap value threshold for now
    nVTIndices = np.array([i for i in range(len(paramList)) if paramList[i].valueThreshold is None], dtype=int)
    ntIterValues, ntIterStdDev, ntRunResultsFound = iterValues[:, :, nVTIndices, :], \
                                                    iterStdDev[:, :, nVTIndices, :], \
                                                    runResultsFound[:, :, nVTIndices]
    nVTParams = [paramList[i] for i in nVTIndices]
    nVTIndicesE = np.array([i for i in range(len(paramListE)) if paramListE[i].valueThreshold is None], dtype=int)
    ntIterValuesE, ntIterStdDevE, ntRunResultsFoundE = iterValuesE[:, :, nVTIndicesE, :], \
                                                    iterStdDevE[:, :, nVTIndicesE, :], \
                                                    runResultsFoundE[:, :, nVTIndicesE]
    nVTParamsE = [paramListE[i] for i in nVTIndicesE]

    # numEnvs*numParams*numIters
    meanVals, meanStdDev, runStdDev, runRange, numValues = \
        np.nanmean(ntIterValues, axis=0), np.nanmean(ntIterStdDev, axis=0), np.nanstd(ntIterValues, axis=0), \
        (np.nanmin(ntIterValues, axis=0), np.nanmax(ntIterValues, axis=0)), np.count_nonzero(ntRunResultsFound, axis=0)
    meanValsE, meanStdDevE, runStdDevE, runRangeE, numValuesE = \
        np.nanmean(ntIterValuesE, axis=0), np.nanmean(ntIterStdDevE, axis=0), np.nanstd(ntIterValuesE, axis=0), \
        (np.nanmin(ntIterValuesE, axis=0), np.nanmax(ntIterValuesE, axis=0)), np.count_nonzero(ntRunResultsFoundE, axis=0)

    # Figure out what the values are for each parameter we're talking about
    numNodeValues = np.unique(np.array([p.numNodes for p in nVTParams], dtype=int))
    numNodeIndices = getParamIndices(numNodeValues, 'numNodes', nVTParams)
    numSampleValues = np.unique(np.array([p.numSamples for p in nVTParams], dtype=int))
    numSampleIndices = getParamIndices(numSampleValues, 'numSamples', nVTParams)
    numBestSampleValues = np.unique(np.array([p.numBestSamples for p in nVTParams], dtype=int))
    numBestSampleIndices = getParamIndices(numBestSampleValues, 'numBestSamples', nVTParams)
    lrValues = np.unique(np.array([p.learningRate for p in nVTParams], dtype=np.float))
    lrIndices = getParamIndices(lrValues, 'learningRate', nVTParams)

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
        plt.legend()
        plt.savefig('Figures/'+envNames[envIndex]+'Nodes.pgf')
        plt.cla()

        # num samples per iteration
        sampleMeanVals = [np.nanmean(meanVals[envIndex, numSampleIndex, :], axis=0) for numSampleIndex in numSampleIndices]  # 1000
        plt.plot(iterations, sampleMeanVals[0], label=str(numSampleValues[0]))
        plt.plot(iterations, sampleMeanVals[1], label=str(numSampleValues[1]))
        plt.plot(iterations, sampleMeanVals[2], label=str(numSampleValues[2]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.legend()
        plt.savefig('Figures/'+envNames[envIndex]+'Samples.pgf')
        plt.cla()

        # num best samples per iteration
        bestSampleMeanVals = [np.nanmean(meanVals[envIndex, numBSampleIndex, :], axis=0) for numBSampleIndex in numBestSampleIndices]  # 1000
        plt.plot(iterations, bestSampleMeanVals[0], label=str(numBestSampleValues[0]))
        plt.plot(iterations, bestSampleMeanVals[1], label=str(numBestSampleValues[1]))
        plt.plot(iterations, bestSampleMeanVals[2], label=str(numBestSampleValues[2]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.legend()
        plt.savefig('Figures/'+envNames[envIndex]+'BestSamples.pgf')
        plt.cla()

        # learning rate
        lrMeanVals = [np.nanmean(meanVals[envIndex, lrSampleIndex, :], axis=0) for lrSampleIndex in lrIndices]  # 1000
        plt.plot(iterations, lrMeanVals[0], label=str(lrValues[0]))
        plt.plot(iterations, lrMeanVals[1], label=str(lrValues[1]))
        plt.plot(iterations, lrMeanVals[2], label=str(lrValues[2]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.legend()
        plt.savefig('Figures/'+envNames[envIndex]+'LearningRate.pgf')
        plt.cla()

        # Entropy
        EntMeanVals = [np.nanmean(meanVals[envIndex, :, :], axis=0), np.nanmean(meanValsE[envIndex, :, :], axis=0)]  # 1000
        plt.plot(iterations, EntMeanVals[0], label='No Entropy')
        plt.plot(iterations, EntMeanVals[1], label='Entropy')
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.legend()
        plt.savefig('Figures/'+envNames[envIndex]+'Entropy.pgf')
        plt.cla()

def extractDPOMDP():
    if os.path.isfile('compiledValuesDPOMDP.npz') and os.path.isfile('compiledValueLabelsDPOMDP.pkl'):
        values = np.load('compiledValuesDPOMDP.npz')
        runResultsFound, iterValues, iterStdDev = tuple(
            values[item] for item in ['runResultsFound', 'iterValue', 'iterStdDev'])
        labels = pickle.load(open('compiledValueLabelsDPOMDP.pkl', 'rb'))
        envNames, paramList = labels['envs'], labels['params']
        values = np.load('compiledValuesDPOMDPEnt.npz')
        runResultsFoundE, iterValuesE, iterStdDevE = tuple(
            values[item] for item in ['runResultsFound', 'iterValue', 'iterStdDev'])
        labelsE = pickle.load(open('compiledValueLabelsDPOMDPEnt.pkl', 'rb'))
        envNamesE, paramListE = labelsE['envs'], labelsE['params']
    else:
        basePath = '/home/david/GDiceRes'
        entPath = '/home/david/GDiceRes/Entropy'
        # runs*envs*params
        # runs*envs*params*iters
        # runs*envs*params*iters
        runResultsFound, iterValues, iterStdDev, envNames, paramList, runDirs = extractResultsFromAllRunsDPOMDP(basePath,True,sepName='compiledValuesDPOMDP.npz', sepName2='compiledValueLabelsDPOMDP.pkl')
        runResultsFoundE, iterValuesE, iterStdDevE, envNamesE, paramListE, runDirsE = extractResultsFromAllRunsDPOMDP(entPath,True, sepName='compiledValuesDPOMDPEnt.npz', sepName2='compiledValueLabelsDPOMDPEnt.pkl')

    # Scrap value threshold for now
    nVTIndices = np.array([i for i in range(len(paramList)) if paramList[i].valueThreshold is None], dtype=int)
    ntIterValues, ntIterStdDev, ntRunResultsFound = iterValues[:, :, nVTIndices, :], \
                                                    iterStdDev[:, :, nVTIndices, :], \
                                                    runResultsFound[:, :, nVTIndices]
    nVTParams = [paramList[i] for i in nVTIndices]
    nVTIndicesE = np.array([i for i in range(len(paramListE)) if paramListE[i].valueThreshold is None], dtype=int)
    ntIterValuesE, ntIterStdDevE, ntRunResultsFoundE = iterValuesE[:, :, nVTIndicesE, :], \
                                                       iterStdDevE[:, :, nVTIndicesE, :], \
                                                       runResultsFoundE[:, :, nVTIndicesE]
    nVTParamsE = [paramListE[i] for i in nVTIndicesE]

    # numEnvs*numParams*numIters
    meanVals, meanStdDev, runStdDev, runRange, numValues = \
        np.nanmean(ntIterValues, axis=0), np.nanmean(ntIterStdDev, axis=0), np.nanstd(ntIterValues, axis=0), \
        (np.nanmin(ntIterValues, axis=0), np.nanmax(ntIterValues, axis=0)), np.count_nonzero(ntRunResultsFound, axis=0)
    meanValsE, meanStdDevE, runStdDevE, runRangeE, numValuesE = \
        np.nanmean(ntIterValuesE, axis=0), np.nanmean(ntIterStdDevE, axis=0), np.nanstd(ntIterValuesE, axis=0), \
        (np.nanmin(ntIterValuesE, axis=0), np.nanmax(ntIterValuesE, axis=0)), np.count_nonzero(ntRunResultsFoundE,
                                                                                               axis=0)

    # Figure out what the values are for each parameter we're talking about
    numNodeValues = np.unique(np.array([p.numNodes for p in nVTParams], dtype=int))
    numNodeIndices = getParamIndices(numNodeValues, 'numNodes', nVTParams)
    numSampleValues = np.unique(np.array([p.numSamples for p in nVTParams], dtype=int))
    numSampleIndices = getParamIndices(numSampleValues, 'numSamples', nVTParams)
    numBestSampleValues = np.unique(np.array([p.numBestSamples for p in nVTParams], dtype=int))
    numBestSampleIndices = getParamIndices(numBestSampleValues, 'numBestSamples', nVTParams)
    lrValues = np.unique(np.array([p.learningRate for p in nVTParams], dtype=np.float))
    lrIndices = getParamIndices(lrValues, 'learningRate', nVTParams)
    ceValues = np.array([False, True], dtype=np.bool)
    ceIndices = getParamIndices(ceValues, 'centralized', nVTParams)

    iterations = np.arange(meanVals.shape[-1])
    os.makedirs('Figures', exist_ok=True)
    for envIndex in range(meanVals.shape[0]):
        # num nodes
        nodeMeanVals = [np.nanmean(meanVals[envIndex, nodeValIndex, :], axis=0) for nodeValIndex in
                        numNodeIndices]  # 1000
        plt.plot(iterations, nodeMeanVals[0], label=str(numNodeValues[0]))
        plt.plot(iterations, nodeMeanVals[1], label=str(numNodeValues[1]))
        plt.plot(iterations, nodeMeanVals[2], label=str(numNodeValues[2]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.legend()
        plt.savefig('Figures/' + envNames[envIndex] + 'Nodes.pgf')
        plt.cla()

        # num samples per iteration
        sampleMeanVals = [np.nanmean(meanVals[envIndex, numSampleIndex, :], axis=0) for numSampleIndex in
                          numSampleIndices]  # 1000
        plt.plot(iterations, sampleMeanVals[0], label=str(numSampleValues[0]))
        plt.plot(iterations, sampleMeanVals[1], label=str(numSampleValues[1]))
        plt.plot(iterations, sampleMeanVals[2], label=str(numSampleValues[2]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.legend()
        plt.savefig('Figures/' + envNames[envIndex] + 'Samples.pgf')
        plt.cla()

        # num best samples per iteration
        bestSampleMeanVals = [np.nanmean(meanVals[envIndex, numBSampleIndex, :], axis=0) for numBSampleIndex in
                              numBestSampleIndices]  # 1000
        plt.plot(iterations, bestSampleMeanVals[0], label=str(numBestSampleValues[0]))
        plt.plot(iterations, bestSampleMeanVals[1], label=str(numBestSampleValues[1]))
        plt.plot(iterations, bestSampleMeanVals[2], label=str(numBestSampleValues[2]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.legend()
        plt.savefig('Figures/' + envNames[envIndex] + 'BestSamples.pgf')
        plt.cla()

        # learning rate
        lrMeanVals = [np.nanmean(meanVals[envIndex, lrSampleIndex, :], axis=0) for lrSampleIndex in lrIndices]  # 1000
        plt.plot(iterations, lrMeanVals[0], label=str(lrValues[0]))
        plt.plot(iterations, lrMeanVals[1], label=str(lrValues[1]))
        plt.plot(iterations, lrMeanVals[2], label=str(lrValues[2]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.legend()
        plt.savefig('Figures/' + envNames[envIndex] + 'LearningRate.pgf')
        plt.cla()

        # centralized
        ceMeanVals = [np.nanmean(meanVals[envIndex, ceSampleIndex, :], axis=0) for ceSampleIndex in ceIndices]  # 1000
        plt.plot(iterations, ceMeanVals[0], label=str(ceValues[0]))
        plt.plot(iterations, ceMeanVals[1], label=str(ceValues[1]))
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.legend()
        plt.savefig('Figures/' + envNames[envIndex] + 'Centralized.pgf')
        plt.cla()

        # Entropy
        EntMeanVals = [np.nanmean(meanVals[envIndex, :, :], axis=0),
                       np.nanmean(meanValsE[envIndex, :, :], axis=0)]  # 1000
        plt.plot(iterations, EntMeanVals[0], label='No Entropy')
        plt.plot(iterations, EntMeanVals[1], label='Entropy')
        plt.xlabel("Iterations")
        plt.ylabel("Performance")
        plt.legend()
        plt.savefig('Figures/' + envNames[envIndex] + 'Entropy.pgf')
        plt.cla()

if __name__ == "__main__":
    extractPOMDP()
    extractDPOMDP()
