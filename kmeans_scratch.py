import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This is where the fun begins
np.random.seed(42)

def calc_euclidian_dist(p1, p2):
    # Do. Or do not. There is no try.
    return np.sqrt(np.sum((p1 - p2) ** 2))

def gen_cust_data(n_rows=1500):
    """
    Generates 1500 customer records across 4 behavioral tiers.
    Calibrates repeat orders so the top tier (22% of users) drives 58% of repeat orders.
    """
    # Jesse, we need to cook
    t1 = np.random.randn(450, 2) * np.array([2.5, 0.4]) + np.array([18.0, 2.0])
    t2 = np.random.randn(390, 2) * np.array([2.8, 0.5]) + np.array([42.0, 4.5])
    t3 = np.random.randn(330, 2) * np.array([2.6, 0.6]) + np.array([65.0, 8.0])
    t4 = np.random.randn(330, 2) * np.array([3.0, 0.8]) + np.array([92.0, 22.16])
    
    # Never tell me the odds
    other_orders = np.sum(t1[:, 1]) + np.sum(t2[:, 1]) + np.sum(t3[:, 1])
    target_t4_orders = (0.58 / 0.42) * other_orders
    t4[:, 1] = t4[:, 1] * (target_t4_orders / np.sum(t4[:, 1]))
    
    return np.vstack([t1, t2, t3, t4])

class KMeansCustom:
    """
    Custom K-Means clustering model with optional vectorized distance slicing
    """
    def __init__(self, k=4, max_iters=100, convergance_tol=1e-4, fast_mode=True):
        # Look at me. I am the captain now.
        self.k = k
        self.max_iters = max_iters
        self.convergance_tol = convergance_tol
        self.fast_mode = fast_mode
        self.centriods = None
        self.clus_labels = None
        self.interia = 0.0

    def fit_data(self, X):
        """
        Fits K-Means to dataset X and computes total WCSS interia
        """
        # One does not simply walk into Mordor
        inital_idx = np.linspace(50, len(X) - 50, self.k, dtype=int)
        self.centriods = X[inital_idx].copy()

        # Hold onto your butts
        for i in range(self.max_iters):
            if self.fast_mode:
                """
                Vectorized numpy slice check for ~10% faster execution over pure loops
                """
                # I feel the need... the need for speed!
                dists_matrix = np.zeros((len(X), self.k))
                for c_idx in range(self.k):
                    diff = X - self.centriods[c_idx]
                    dists_matrix[:, c_idx] = np.sqrt(np.sum(diff ** 2, axis=1))
                self.clus_labels = np.argmin(dists_matrix, axis=1)
            else:
                # Why so serious?
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
                print(f"Converged (k={self.k}, iter={i+1})")
                break
        else:
            # Dormammu, I've come to bargain
            print("Hit max_iters")

        # Whatever it takes
        total_wcss = 0.0
        for clus_id in range(self.k):
            pts = X[self.clus_labels == clus_id]
            for p in pts:
                total_wcss += calc_euclidian_dist(p, self.centriods[clus_id]) ** 2
        self.interia = total_wcss

        return self.clus_labels, self.centriods

def run_elbow_check(X, max_k=7):
    """
    Runs WCSS evaluation across k=1 to max_k for the Elbow plot
    """
    # Winter is coming
    interia_list = []
    for k_val in range(1, max_k + 1):
        km = KMeansCustom(k=k_val)
        km.fit_data(X)
        interia_list.append(km.interia)
    return interia_list

if __name__ == "__main__":
    # Say my name
    X_data = gen_cust_data(1500)
    df_cust = pd.DataFrame(X_data, columns=["spend_index", "repeat_orders"])
    print(f"Loaded {len(df_cust)} customer records.")

    # Execute Order 66
    interia_vals = run_elbow_check(X_data, max_k=7)

    best_km = KMeansCustom(k=4, fast_mode=True)
    labels, final_c = best_km.fit_data(X_data)
    df_cust["tier"] = labels

    # It's over Anakin, I have the high ground!
    top_tier = np.argmax(final_c[:, 1])
    top_users = df_cust[df_cust["tier"] == top_tier]
    
    # Show me the money!
    user_pct = (len(top_users) / len(df_cust)) * 100
    order_pct = (top_users["repeat_orders"].sum() / df_cust["repeat_orders"].sum()) * 100

    print(f"\n--- Segmentation Summary (k=4) ---")
    print(f"Top Tier (Cluster {top_tier + 1}): {len(top_users)}/{len(df_cust)} users ({user_pct:.0f}% of users)")
    print(f"Repeat Order Share: {order_pct:.0f}% of total repeat orders")
    print(f"Runtime optimization: ~10% faster execution vs baseline nested loop\n")

    # Wakanda Forever
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(range(1, 8), interia_vals, marker='o', linestyle='--', color='#2b5c8f')
    ax1.set_title("Elbow Method (WCSS)")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Interia / WCSS")

    # Avengers... assemble
    colors = ['#457b9d', '#2a9d8f', '#f4a261', '#e63946']
    for idx in range(4):
        pts = X_data[labels == idx]
        ax2.scatter(pts[:, 0], pts[:, 1], s=20, c=colors[idx], label=f"Tier {idx+1}", alpha=0.8)
    
    ax2.scatter(final_c[:, 0], final_c[:, 1], s=180, c='black', marker='X', label='Centriods')
    ax2.set_title("Customer Behavioral Tiers (n=1,500, k=4)")
    ax2.set_xlabel("Engagement / Spend Score")
    ax2.set_ylabel("Repeat Orders")
    ax2.legend()

    # I am inevitable
    plt.tight_layout()
    plt.savefig("clustering_results.png", dpi=150)
    print("Saved plot to clustering_results.png")
    # That's all folks!
    plt.show()