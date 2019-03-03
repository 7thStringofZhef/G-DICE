import gym
import os
import argparse
import sys
from gym_dpomdps import list_dpomdps
from gym_pomdps import list_pomdps
from multiprocessing import Pool
from GDICE_Python.Parameters import GDICEParams
from GDICE_Python.Controllers import FiniteStateControllerDistribution, DeterministicFiniteStateController
from GDICE_Python.Algorithms import runGDICEOnEnvironment
from GDICE_Python.Scripts import getGridSearchGDICEParams, saveResults, loadResults, checkIfFinished, checkIfPartial, claimRunEnvParamSet, registerRunEnvParamSetCompletion, claimRunEnvParamSet_unfinished, registerRunEnvParamSetCompletion_unfinished
import glob

def runOnListFileDPOMDP(baseSavePath, listFilePath='DPOMDPsToEval.txt', injectEntropy=False):
    # For now, can't go back to inprogress ones
    pool = Pool()
    pString = claimRunEnvParamSet(listFilePath)
    while pString is not None:
        splitPString = pString.split('/')  # {run}/{env}/{param}
        run = splitPString[0]
        os.makedirs(os.path.join(baseSavePath, run), exist_ok=True)
        envName = splitPString[1]
        params = GDICEParams().fromName(name=splitPString[2])
        # Check if run was already finished
        if checkIfFinished(envName, params.name, baseDir=os.path.join(baseSavePath, run))[0]:
            print(params.name + ' already finished for ' + envName + ', skipping...', file=sys.stderr)
            # Remove from in progress
            registerRunEnvParamSetCompletion(pString, listFilePath)
            # Claim next one
            pString = claimRunEnvParamSet(listFilePath)
            continue

        # Check if the run was partially finished
        wasPartiallyRun, npzFilename = checkIfPartial(envName, params.name, baseDir=os.path.join(baseSavePath, run))
        try:
            env = gym.make(envName)
        except MemoryError:
            print(envName + ' too large for memory', file=sys.stderr)
            return
        except Exception as e:
            print(envName + ' encountered error in creation', file=sys.stderr)
            print(e, file=sys.stderr)
            return

        prevResults = None
        if wasPartiallyRun:
            print(params.name + ' partially finished for ' + envName + ', loading...', file=sys.stderr)
            prevResults, FSCDist = loadResults(npzFilename)[:2]
        else:
            if params.centralized:
                FSCDist = FiniteStateControllerDistribution(params.numNodes, env.action_space[0].n,
                                                            env.observation_space[0].n, shouldInjectNoiseUsingMaximalEntropy=injectEntropy)
            else:
                FSCDist = [FiniteStateControllerDistribution(params.numNodes, env.action_space[a].n,
                                                             env.observation_space[a].n, shouldInjectNoiseUsingMaximalEntropy=injectEntropy) for a in range(env.agents)]
        prevResults = None
        env.reset()
        try:
            results = runGDICEOnEnvironment(env, FSCDist, params, parallel=pool, results=prevResults, baseDir=os.path.join(baseSavePath, run))
        except MemoryError:
            print(envName + ' too large for parallel processing. Switching to MultiEnv...', file=sys.stderr)
            results = runGDICEOnEnvironment(env, FSCDist, params, parallel=None, results=prevResults, baseDir=os.path.join(baseSavePath, run))
        except Exception as e:
            print(envName + ' encountered error in runnning' + params.name + ', skipping to next param', file=sys.stderr)
            print(e, file=sys.stderr)
            return
        saveResults(os.path.join(os.path.join(baseSavePath, run), 'EndResults'), envName, params, results)

        # Remove from in progress
        registerRunEnvParamSetCompletion(pString, listFilePath)
        # Delete the temp results
        try:
            for filename in glob.glob(os.path.join(os.path.join(baseSavePath, run), 'GDICEResults', envName, params.name) + '*'):
                os.remove(filename)
        except:
            return

        # Claim next one
        pString = claimRunEnvParamSet(listFilePath)

def runOnListFile_unfinished(baseSavePath, listFilePath='POMDPsToEval.txt', injectEntropy=True):
    pool = Pool()
    pString = claimRunEnvParamSet_unfinished(listFilePath)
    while pString is not None:
        splitPString = pString.split('/')  # {run}/{env}/{param}
        run = splitPString[0]
        os.makedirs(os.path.join(baseSavePath, run), exist_ok=True)
        envName = splitPString[1]
        params = GDICEParams().fromName(name=splitPString[2])
        try:
            env = gym.make(envName)
        except MemoryError:
            print(envName + ' too large for memory', file=sys.stderr)
            return
        except Exception as e:
            print(envName + ' encountered error in creation', file=sys.stderr)
            print(e, file=sys.stderr)
            return

        wasPartiallyRun, npzFilename = checkIfPartial(envName, params.name)
        prevResults = None
        if wasPartiallyRun:
            print(params.name + ' partially finished for ' + envName + ', loading...', file=sys.stderr)
            prevResults, FSCDist = loadResults(npzFilename)[:2]
        else:
            FSCDist = FiniteStateControllerDistribution(params.numNodes, env.action_space.n,
                                                        env.observation_space.n)
        env.reset()
        try:
            results = runGDICEOnEnvironment(env, FSCDist, params, parallel=pool, results=prevResults, baseDir=os.path.join(baseSavePath, run))
        except MemoryError:
            print(envName + ' too large for parallel processing. Switching to MultiEnv...', file=sys.stderr)
            results = runGDICEOnEnvironment(env, FSCDist, params, parallel=None, results=prevResults, baseDir=os.path.join(baseSavePath, run))
        except Exception as e:
            print(envName + ' encountered error in runnning' + params.name + ', skipping to next param', file=sys.stderr)
            print(e, file=sys.stderr)
            return
        saveResults(os.path.join(os.path.join(baseSavePath, run), 'EndResults'), envName, params, results)

        # Remove from in progress
        registerRunEnvParamSetCompletion(pString, listFilePath)
        # Delete the temp results
        try:
            for filename in glob.glob(os.path.join(os.path.join(baseSavePath, run), 'GDICEResults', envName, params.name) + '*'):
                os.remove(filename)
        except:
            return

        # Claim next one
        pString = claimRunEnvParamSet_unfinished(listFilePath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose save dir and environment')
    parser.add_argument('--save_path', type=str, default='/scratch/slayback.d/GDICE', help='Base save path')
    parser.add_argument('--env_name', type=str, default='', help='Environment to run')
    parser.add_argument('--env_type', type=str, default='POMDP', help='Environment type to run')
    parser.add_argument('--set_list', type=str, default='', help='If provided, uses a list of run/env/param sets instead')
    parser.add_argument('--unfinished', type=int, default=0, help='If 1, clean out unfinished results')
    args = parser.parse_args()
    if not args.set_list:
        pass
    else:
        useEntropy = False
        if args.unfinished:
            runFn = runOnListFile_unfinished if args.env_type == 'POMDP' else runOnListFileDPOMDP_unfinished
        else:
            runFn = runOnListFile if args.env_type =='POMDP' else runOnListFileDPOMDP
        if args.set_list.startswith('Ent'):
            useEntropy = True
        runFn(args.save_path, args.set_list, injectEntropy=useEntropy)