from GDICE_Python.Plotting import *
from GDICE_Python.Scripts import extractResultsFromAllRuns
import os
import pickle
import numpy as np

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

    # numEnvs*numParams*numIters
    meanVals, meanStdDev, runStdDev, runRange = \
        np.nanmean(iterValues, axis=0), np.nanmean(iterStdDev, axis=0), np.nanstd(iterValues, axis=0), \
        (np.nanmin(iterValues, axis=0), np.nanmax(iterValues, axis=0))


    pass