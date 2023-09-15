import numpy as np
import time
from same_seeds import same_seeds
same_seeds(0)
np.set_printoptions(suppress=True, precision=5)


def QR_inv(m, mode='householder'):
    eps = 1e-3
    A = np.mat(m.copy(), dtype=float)
    dim = A.shape[0]
    Q = np.mat(A, dtype=float)
    R = np.mat(np.zeros([dim, dim]), dtype=float)

    for j in range(dim):
        for i in range(j):
            R[i, j] = (Q[:, i].T * A[:, j]).item()
            for k in range(dim):
                Q[k, j] -= (R[i, j] * Q[k, i])

        t = 0
        for i in range(dim):
            t += Q[i, j] ** 2

        R[j, j] = np.sqrt(t)

        for i in range(dim):
            Q[i, j] /= R[j, j]

    # 求逆
    E = np.mat(np.eye(dim), dtype=float)  # E用来求R的逆
    for i in range(dim - 1, -1, -1):
        if abs(R[i, i] - 1.0) > eps:
            for k in range(i, dim):
                # 对角元单位化
                E[i, k] = E[i, k] / R[i, i]
        for j in range(i-1, -1, -1):
            for k in range(j+1, dim):
                E[j, k] = E[j, k] - R[j, i] * E[i, k]

    inv = E * Q.T
    return np.array(inv)


if __name__ == '__main__':
    A = np.array([[10, 7, 8, 7],
                  [7, 5, 6, 5],
                  [8, 6, 10, 9],
                  [7, 5, 9, 10]], dtype='float')
    X = QR_inv(A)
    print(X @ A)
    A = np.random.randn(50,50)


    # start = time.time()
    # for i in range(1000):
    #     X = QR_inv(A)
    # print(time.time() - start)