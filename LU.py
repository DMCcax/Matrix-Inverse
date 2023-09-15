import numpy as np
from same_seeds import same_seeds

same_seeds(0)
np.set_printoptions(suppress=True, precision=5)


def LU_inv(m):
    eps = 1e-3
    dim = m.shape[0]
    L = np.mat(np.eye(dim))
    U = np.mat(m.copy())

    for i in range(dim):
        if abs(U[i, i]) < eps:
            print("LU: There is no non-zero pivot!")
            return np.zeros(1)
        for j in range(i+1, dim):
            if abs(U[i, i] - 1.0) < eps:
                L[j, i] = U[j, i]
            else:
                L[j, i] = U[j, i] / U[i, i]
            U[j, i] = 0
            for k in range(i+1, dim):
                U[j, k] = U[j, k] - L[j, i] * U[i, k]


    E1 = np.mat(np.eye(dim))  # 这个E1用来求U的逆
    for i in range(dim - 1, -1, -1):
        if abs(U[i, i] - 1.0) > eps:
            for k in range(i, dim):
                # 对角元单位化
                E1[i, k] = E1[i, k] / U[i, i]
                # U[i, :] = U[i, :] / U[i, i]
        for j in range(i-1, -1, -1):
            for k in range(j+1, dim):
                E1[j, k] = E1[j, k] - U[j, i] * E1[i, k]
                # U[:i, :] = U[:i, :] - U[:i, i] * U[i, :]


    # 当然，我们还可以来求一下下三角阵L的逆
    E2 = np.mat(np.eye(dim))
    for i in range(dim):
        for j in range(i+1, dim):
            for k in range(0, j+1):
                E2[j, k] = E2[j, k] - L[j, i] * E2[i, k]
                # L[i + 1:, :] = L[i + 1:, :] - L[i + 1:, i] * L[i, :]

    return np.array(E1 * E2)


if __name__ == "__main__":
    # A = np.mat(np.random.randn(3, 3))
    A = np.array([[1, 2, 3],  # for test
                  [3, 5, 2],
                  [1, 0, 1]])
    X = LU_inv(A)
    print(A @ X)
