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
from GDICE_Python.Scripts import getGridSearchGDICEParams, getGridSearchGDICEParamsDPOMDP, checkIfPartial, checkIfFinished
import glob

if __name__ == "__main__":
    baseDir = '/scratch/slayback.d/GDICE'
    baseEntDir = '/scratch/slayback.d/GDICE/Entropy'
    envStringsP, paramListP = getGridSearchGDICEParams()
    envStringsDP, paramListDP = getGridSearchGDICEParamsDPOMDP()
    runs = ['1', '2']
    poms, dpoms, entpoms, entdpoms = [], [], [], []
    for run in runs:
        for env in envStringsP:
            for param in paramListP:
                if not checkIfFinished(env, param.name, baseDir=os.path.join(baseDir, run))[0]:
                    poms.append(os.path.join(run, env, param.name))
                if not checkIfFinished(env, param.name, baseDir=os.path.join(baseEntDir, run))[0]:
                    entpoms.append(os.path.join(run, env, param.name))
        for env in envStringsDP:
            for param in paramListDP:
                if not checkIfFinished(env, param.name, baseDir=os.path.join(baseDir, run))[0]:
                    dpoms.append(os.path.join(run, env, param.name))
                if not checkIfFinished(env, param.name, baseDir=os.path.join(baseEntDir, run))[0]:
                    entdpoms.append(os.path.join(run, env, param.name))

    with open('POMDPsToEval.txt', 'w') as f:
        for fileStr in poms:
            f.write(fileStr+'\n')

    with open('DPOMDPsToEval.txt', 'w') as f:
        for fileStr in dpoms:
            f.write(fileStr+'\n')

    with open('EntPOMDPsToEval.txt', 'w') as f:
        for fileStr in entpoms:
            f.write(fileStr+'\n')

    with open('EntDPOMDPsToEval.txt', 'w') as f:
        for fileStr in poms:
            f.write(fileStr+'\n')

