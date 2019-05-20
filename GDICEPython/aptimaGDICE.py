import gym
import os
import types
import argparse
import sys
from gym_dpomdps import DPOMDP, MultiDPOMDP
from multiprocessing import Pool
from GDICE_Python.Parameters import GDICEParams
from GDICE_Python.Controllers import FiniteStateControllerDistribution, DeterministicFiniteStateController
from GDICE_Python.Algorithms import runGDICEOnEnvironment
from GDICE_Python.Scripts import getGridSearchGDICEParams, saveResults, loadResults, checkIfFinished, checkIfPartial, claimRunEnvParamSet, registerRunEnvParamSetCompletion, claimRunEnvParamSet_unfinished, registerRunEnvParamSetCompletion_unfinished
import glob

# Parameters I'll use. Keeping permutations to a minimum
nNodes = [10, 20]
nSamples = 70
nBestSamples = 5
N_k = 1000
N_sim = 1000
lr = [0.1, 0.2, 0.5]
timeHorizon = 100
injectEntropy = True
baseSavePath = ''
aptimaPath = '/scratch/slayback.d/GDICE/G-DICE'


if __name__ == "__main__":
    # Should probably split runs, envs, and param sets across jobs
    # So I should probably just do the "list" approach again...
    parser = argparse.ArgumentParser(description='Choose save dir and environment')
    parser.add_argument('--save_path', type=str, default='/scratch/slayback.d/GDICE', help='Base save path')
    parser.add_argument('--env_num', type=int, default=0, help='Environment to run (of 3)')
    parser.add_argument('--param_num', type=int, default=0, help='Paramset to run (of 6)')
    parser.add_argument('--run_num', type=int, default=1, help='If provided, uses a list of run/env/param sets instead')
    args = parser.parse_args()
    baseSavePath = args.save_path

    pool = Pool()
    dpomdpsnames = [os.path.join(aptimaPath, name) for name in os.listdir(aptimaPath) if name.endswith('.dpomdp')]  # Get environments
    dpomdps = [DPOMDP(name) for name in dpomdpsnames]
    paramList = [GDICEParams(n, N_k, nSamples, N_sim, N_k, l, None, timeHorizon) for n in nNodes for l in lr]  # Mini grid search
    run = str(args.run_num)
    envName = dpomdpsnames[args.env_num]
    env = DPOMDP(envName)
    actualName = (os.path.splitext(os.path.split(envName)[1])[0]).replace('.', '_')
    env.spec = types.SimpleNamespace()
    env.spec.id = actualName
    env.reset()
    paramSet = paramList[args.param_num]
    FSCDist = FiniteStateControllerDistribution(paramSet.numNodes, env.action_space[0].n, env.observation_space[0].n, injectEntropy)
    results = runGDICEOnEnvironment(env, FSCDist, paramSet, parallel=pool, results=None,
                                    baseDir=os.path.join(baseSavePath, run), saveFrequency=25)
    saveResults(os.path.join(baseSavePath, 'EndResults', run), actualName, paramSet, results)
    # Delete the temp results
    try:
        for filename in glob.glob(os.path.join(baseSavePath, run, 'GDICEResults', actualName, paramSet.name) + '*'):
            os.remove(filename)
    except:
        pass
