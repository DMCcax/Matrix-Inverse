import numpy as np
import time
from same_seeds import same_seeds
same_seeds(0)
np.set_printoptions(suppress=True, precision=5)


def QR_inv(A, mode='householder'):
    eps = 1e-3
    dim, n = A.shape
    Q = np.array(A, dtype=float)
    R = np.zeros((dim, dim), dtype=float)

    for j in range(dim):
        for i in range(j):
            R[i][j] = np.dot(Q[:, i].T, A[:, j])
            Q[:, j] -= (R[i][j] * Q[:, i])

        R[j][j] = np.linalg.norm(Q[:, j])
        Q[:, j] /= R[j][j]

    # 求逆
    Q = np.mat(Q)
    R = np.mat(R)
    E = np.mat(np.eye(dim))  # E用来求R的逆
    for i in range(dim - 1, -1, -1):
        if abs(R[i, i]) > eps:
            if (abs(R[i, i]-1)) > eps:
                # 对角元单位化
                E[i, :] = E[i, :] / R[i, i]
                R[i, :] = R[i, :] / R[i, i]

            E[:i, :] = E[:i, :] - R[:i, i] * E[i, :]
            R[:i, :] = R[:i, :] - R[:i, i] * R[i, :]

    inv = E * Q.T
    return np.array(inv)


if __name__ == '__main__':
    A = np.array([[10, 7, 8, 7],
                  [7, 5, 6, 5],
                  [8, 6, 10, 9],
                  [7, 5, 9, 10]], dtype='float')
    # X = QR_1(A)
    # print(X @ A)
    # A = np.random.randn(50,50)


    start = time.time()
    for i in range(1000):
        X = QR_inv(A)
    print(time.time() - start)