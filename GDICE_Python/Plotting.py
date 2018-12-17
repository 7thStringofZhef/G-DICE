import matplotlib.pyplot as plt
import numpy as np


# Plot the values given by a single run
# Inputs:
#    bestValueAtEachIteration: Loaded (numIters,) nparray of best value at each iteration
#    prevPlot: If None, make a new plot. Otherwise, should be a tuple of Figure and Axes object handles to add this line to a previous plot
#    bestStdDevAtEachIteration: If not none, (numIters,) nparray of the value of the best policy at each iteration within the environment
#    plotFreq: The frequency of iterations to plot. If 1, plots each; if 2, plots every other, etc
def plotSingleRunResults(bestValueAtEachIteration, prevPlot=None, bestStdDevAtEachIteration=None, plotFreq=1):
    # Add to previous plot
    if prevPlot is not None:
        fig, ax = prevPlot[0], prevPlot[1]
    else:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    xIndices = np.arange(0, bestValueAtEachIteration.shape[0], plotFreq)
    ax.plot(xIndices, bestValueAtEachIteration[::plotFreq])
    if bestStdDevAtEachIteration is not None:
        ax.fill_between(xIndices, bestValueAtEachIteration[::plotFreq]+bestStdDevAtEachIteration[::plotFreq],
                        bestValueAtEachIteration[::plotFreq] - bestStdDevAtEachIteration[::plotFreq])

    return fig, ax

if __name__ == "__main__":
    testValues1 = np.random.rand(1000)*50
    testErrs = np.random.randn(1000)
    fig, ax = plotSingleRunResults(testValues1, None, testErrs, 50)
    plt.savefig('test.png')
    pass
