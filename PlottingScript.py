from GDICE_Python.Plotting import *
from GDICE_Python.Scripts import extractResultsFromAllRuns

if __name__ == "__main__":
    basePath = '/media/david/USB STICK/EndResCombined'
    extractResultsFromAllRuns(basePath, True)