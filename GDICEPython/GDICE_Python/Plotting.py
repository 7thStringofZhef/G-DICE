import matplotlib.pyplot as plt
import numpy as np


# Plot the values given by a single run
# Inputs:
#    bestValueAtEachIteration: Loaded (numIters,) nparray of best value at each iteration
#    prevPlot: If None, make a new plot. Otherwise, should be a tuple of Figure and Axes object handles to add this line to a previous plot
#    bestStdDevAtEachIteration: If not none, (numIters,) nparray of the value of the best policy at each iteration within the environment
#    plotFreq: The frequency of iterations to plot. If 1, plots each; if 2, plots every other, etc
# Outputs:
#    fig: Figure object used
#    ax: Axes object used
#    line: Handle specifically for the line object (not the error fill if there is one)
def plotSingleRunResults(bestValueAtEachIteration, prevPlot=None, bestStdDevAtEachIteration=None, plotFreq=1):
    # Add to previous plot
    if prevPlot is not None:
        fig, ax = prevPlot[0], prevPlot[1]
    else:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    xIndices = np.arange(0, bestValueAtEachIteration.shape[0], plotFreq)
    line, = ax.plot(xIndices, bestValueAtEachIteration[::plotFreq])
    if bestStdDevAtEachIteration is not None:
        ax.fill_between(xIndices, bestValueAtEachIteration[::plotFreq]+bestStdDevAtEachIteration[::plotFreq],
                        bestValueAtEachIteration[::plotFreq] - bestStdDevAtEachIteration[::plotFreq])

    return fig, ax, line

# Label a plot with parameters and environment name
def labelAxis(ax, lines, envName, legendNameList):
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value (unnormalized)')
    ax.set_title(envName + ' GDICE run')
    ax.legend(handles=lines, labels=legendNameList)


if __name__ == "__main__":
    testValues1 = np.random.rand(1000)*50
    testErrs = np.random.randn(1000)
    fig, ax, line = plotSingleRunResults(testValues1, None, testErrs, 50)
    fig, ax, line2 = plotSingleRunResults(testValues1*2, (fig,ax), testErrs, 50)
    lines = [line, line2]
    labelAxis(ax, lines, 'POMDP', ['test', 'x2'])
    plt.savefig('test.png')
    pass
