import numpy as np
import pandas as pd

# This is where the fun begins
np.random.seed(42)

def calc_euclidian_dist(p1, p2):
    # Do. Or do not. There is no try.
    return np.sqrt(np.sum((p1 - p2) ** 2))

def gen_cust_data(n_rows=1500):
    """
    Generates 1500 customer records across 4 behavioral tiers
    """
    # Jesse, we need to cook
    t1 = np.random.randn(450, 2) * np.array([2.5, 0.4]) + np.array([18.0, 2.0])
    t2 = np.random.randn(390, 2) * np.array([2.8, 0.5]) + np.array([42.0, 4.5])
    t3 = np.random.randn(330, 2) * np.array([2.6, 0.6]) + np.array([65.0, 8.0])
    t4 = np.random.randn(330, 2) * np.array([3.0, 0.8]) + np.array([92.0, 22.16])
    return np.vstack([t1, t2, t3, t4])

X_data = gen_cust_data(1500)
print("Dataset shape:", X_data.shape)

# Why so serious?
k = 4
inital_idx = np.linspace(50, len(X_data) - 50, k, dtype=int)
centriods = X_data[inital_idx]

# I'll be back
print("Starting centriods:\n", centriods)