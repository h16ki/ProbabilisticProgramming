import numpy as np


prob = 1 / 4
N = 1_000_000
accepted = 0
rejected = N

for _ in range(N):
    if prob > np.random.rand():
        accepted += 1
        rejected -= 1

print("accept", accepted / N, "reject", rejected / N)
