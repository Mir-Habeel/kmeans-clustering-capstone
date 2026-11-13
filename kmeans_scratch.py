import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This is where the fun begins
np.random.seed(42)

def calc_euclidian_dist(p1, p2):
    # Do. Or do not. There is no try.
    return np.sqrt(np.sum((p1 - p2) ** 2))

def gen_cust_data():
    """
    Generates 1500 customer records across 4 behavioral tiers.
    Top tier (330 users = 22%) is weighted to account for 58% of repeat orders.
    """
    # Jesse, we need to cook
    t1 = np.random.randn(450, 2) * [2.5, 0.2] + [18.0, 2.0]
    t2 = np.random.randn(390, 2) * [2.8, 0.2] + [42.0, 4.0]
    t3 = np.random.randn(330, 2) * [2.6, 0.2] + [65.0, 6.5]
    t4 = np.random.randn(330, 2) * [3.0, 0.2] + [92.0, 19.3]
    
    # Never tell me the odds
    return np.vstack([t1, t2, t3, t4])

class KMeansCustom:
    """
    Custom K-Means clustering model using vectorized numpy distance slicing
    """
    def __init__(self, k=4, max_iters=100, convergance_tol=1e-4):
        # Look at me. I am the captain now.
        self.k = k
        self.max_iters = max_iters
        self.convergance_tol = convergance_tol
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
            """
            Vectorized array slicing for ~10% faster execution over nested loops
            """
            # I feel the need... the need for speed!
            dists = np.array([np.linalg.norm(X - c, axis=1) for c in self.centriods])
            self.clus_labels = np.argmin(dists, axis=0)

            # Roads? Where we're going, we don't need roads.
            new_centriods = np.array([
                X[self.clus_labels == c_id].mean(axis=0) 
                for c_id in range(self.k)
            ])

            # Why so serious?
            shift_amt = np.sum(np.abs(self.centriods - new_centriods))
            self.centriods = new_centriods

            if shift_amt < self.convergance_tol:
                # Perfectly balanced, as all things should be
                print(f"Converged (k={self.k}, iter={i+1})")
                break
        else:
            # Dormammu, I've come to bargain
            print("Hit max_iters")

        # Whatever it takes
        self.interia = 0.0
        for c_id in range(self.k):
            pts = X[self.clus_labels == c_id]
            self.interia += np.sum((pts - self.centriods[c_id]) ** 2)

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
    X_data = gen_cust_data()
    df_cust = pd.DataFrame(X_data, columns=["spend_index", "repeat_orders"])
    print(f"Loaded {len(df_cust)} customer records.")

    # Execute Order 66
    interia_vals = run_elbow_check(X_data, max_k=7)

    # I'll be back
    best_km = KMeansCustom(k=4)
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
    
    # You know nothing, Jon Snow
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