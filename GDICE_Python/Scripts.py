import numpy as np

N_n = np.arange(5, 16)  # 5-16 nodes for FSC
N_k = np.arange(20, 51)  # 20-50 iterations
N_s = np.arange(30, 71, 5)  # 30-70 samples per iteration (by 5)
N_b = np.arange(3, 10)  # 3-8 best samples to keep
N_sim = np.arange(200, 1001, 100)  # 200-1000 simulations per sample (by 100)
lr = np.arange(0.05, 0.31, 0.05)  # .05-.3 learning rate (by 0.05)

