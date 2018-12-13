import numpy as np
import gym
from gym_pomdps import list_pomdps

def gridSeachGDICE():
    N_n = np.arange(5, 16)  # 5-16 nodes for FSC
    N_k = np.arange(20, 51, 5)  # 20-50 iterations
    N_s = np.arange(30, 71, 5)  # 30-70 samples per iteration (by 5)
    N_b = np.arange(3, 10)  # 3-9 best samples to keep
    N_sim = np.arange(200, 1001, 100)  # 200-1000 simulations per sample (by 100)
    lr = np.arange(0.05, 0.31, 0.05)  # .05-.3 learning rate (by 0.05)
    vThresholds = [None, 0]  # Either no threshold or no non-negative values

    # All registered Pomdp environments, only the episodic versions
    envStrings = [pomdp for pomdp in list_pomdps() if 'episodic' in pomdp]
    for envStr in envStrings:
        try:
            gym.make(envStr)
        except MemoryError:
            print(envStr +' caused memory error')
            continue

if __name__ == "__main__":
    gridSeachGDICE()