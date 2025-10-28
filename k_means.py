import numpy as np

data = np.array([2, 4, 10, 12, 3, 20, 30, 11, 25])
print(f"Initial Data Set: {data}\n")

m1 = 3
m2 = 4
print("Step 1: Initial Means Assigned")
print(f"Initial m1 (mean of k1): {m1}")
print(f"Initial m2 (mean of k2): {m2}\n")

k1 = []
k2 = []

def distance(x, mean):
    return abs(x - mean)

iteration = 1
max_iterations = 10

while iteration <= max_iterations:
    print(f"--- Iteration {iteration} ---")

    new_k1 = []
    new_k2 = []
    for x in data:
        dist_to_m1 = distance(x, m1)
        dist_to_m2 = distance(x, m2)
        if dist_to_m1 <= dist_to_m2:
            new_k1.append(x)
        else:
            new_k2.append(x)

    new_k1 = np.array(sorted(new_k1))
    new_k2 = np.array(sorted(new_k2))

    if np.array_equal(new_k1, k1) and np.array_equal(new_k2, k2):
        print("Step 8 (Stop): Clusters in this step are the same as the previous step. Convergence achieved.")
        break

    k1 = new_k1
    k2 = new_k2

    print(f"Step {iteration+1} (Assign Clusters):")
    print(f"k1 (clustered around {m1}): {k1}")
    print(f"k2 (clustered around {m2}): {k2}")

    m1_new = k1.mean() if k1.size > 0 else m1
    m2_new = k2.mean() if k2.size > 0 else m2

    print(f"Step {iteration+1} (Update Means):")
    print(f"New m1: {m1_new:.2f}")
    print(f"New m2: {m2_new:.2f}\n")

    m1 = m1_new
    m2 = m2_new
    iteration += 1

print("\n" + "="*40)
print("Final Answer (K-means Output):")
print(f"Cluster k1: {k1}")
print(f"Cluster k2: {k2}")
print("="*40)
