import numpy as np
import gym
from gym_pomdps import list_pomdps
from multiprocessing import Pool
from GDICE_Python.Algorithms import GDICEParams


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
    envStrings = [pomdp for pomdp in list_pomdps() if 'episodic' in pomdp and 'rock' not in pomdp]
    paramList = [GDICEParams(n, N_k, j, N_sim, k, l, t, timeHorizon) for n in N_n for j in N_s for k in N_b for l in lr for t in vThresholds]
    return envStrings, paramList

