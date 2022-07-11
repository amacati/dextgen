import time

from envs.rotations import euler2quat, mat2quat, fastmat2quat, quat2mat
import numpy as np

N = 1000
tnormal = tfast = 0

for _ in range(N):
    euler = 2 * (np.random.rand(256, 3) - 0.5) * np.array([np.pi, np.pi, np.pi / 2])
    mat = quat2mat(euler2quat(euler))

    t1 = time.perf_counter()
    qnormal = mat2quat(mat)
    t2 = time.perf_counter()

    t3 = time.perf_counter()
    qfast = fastmat2quat(mat)
    t4 = time.perf_counter()

    tnormal += t2 - t1
    tfast += t4 - t3

    if np.max(np.abs(qnormal - qfast)) > 1e-9:
        print(np.max(np.abs(qnormal - qfast)))

print(f"tnormal: {tnormal/N:e}, tfast: {tfast/N:e}")
