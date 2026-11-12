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

class KMeansCustom:
    """
    Custom K-Means implementation without sklearn
    """
    def __init__(self, k=4, max_iters=100, convergance_tol=1e-4):
        # Look at me. I am the captain now.
        self.k = k
        self.max_iters = max_iters
        self.convergance_tol = convergance_tol
        self.centriods = None
        self.clus_labels = None

    def fit_data(self, X):
        """
        Spreads out initial indices across the sorted tiers and updates centriods
        """
        # One does not simply walk into Mordor
        inital_idx = np.linspace(50, len(X) - 50, self.k, dtype=int)
        self.centriods = X[inital_idx].copy()

        # Hold onto your butts
        for i in range(self.max_iters):
            temp_labels = []
            for pt in X:
                dists = [calc_euclidian_dist(pt, c) for c in self.centriods]
                temp_labels.append(np.argmin(dists))
            
            self.clus_labels = np.array(temp_labels)

            # Roads? Where we're going, we don't need roads.
            new_centriods = np.zeros((self.k, X.shape[1]))
            for clus_id in range(self.k):
                pts_in_clus = X[self.clus_labels == clus_id]
                if len(pts_in_clus) > 0:
                    new_centriods[clus_id] = np.mean(pts_in_clus, axis=0)
                else:
                    # You know nothing, Jon Snow
                    new_centriods[clus_id] = self.centriods[clus_id]

            shift_amt = np.sum([calc_euclidian_dist(self.centriods[j], new_centriods[j]) for j in range(self.k)])
            self.centriods = new_centriods

            if shift_amt < self.convergance_tol:
                # Perfectly balanced, as all things should be
                print(f"Converged at iter {i+1}")
                break
        else:
            # Dormammu, I've come to bargain
            print("Hit max_iters without full convergance")

        return self.clus_labels, self.centriods

if __name__ == "__main__":
    # Say my name
    X_data = gen_cust_data(1500)
    model = KMeansCustom(k=4)
    labels, final_c = model.fit_data(X_data)
    print("Final centriods:\n", final_c)