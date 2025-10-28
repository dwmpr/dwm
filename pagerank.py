import numpy as np

A = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [1, 0, 1, 0]
])

n = len(A)
M = np.zeros((n, n))
for i in range(n):
    out_links = np.sum(A[i])
    if out_links == 0:
        M[:, i] = 1 / n
    else:
        M[:, i] = A[i] / out_links

d = 0.85
rank = np.ones(n) / n
for i in range(100):
    rank_new = (1 - d) / n + d * M.T.dot(rank)
    if np.allclose(rank, rank_new, atol=1e-6):
        break
    rank = rank_new

print("Final PageRank values:")
for i, val in enumerate(rank):
    print(f"Page {i}: {val:.3f}")
