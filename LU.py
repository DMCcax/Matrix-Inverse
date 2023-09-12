import numpy as np
from same_seeds import same_seeds

same_seeds(0)
np.set_printoptions(suppress=True, precision=5)


def LU_inv(m):
    eps = 1e-3
    n = m.shape[0]
    L = np.mat(np.eye(n))
    U = np.mat(m.copy())
    for i in range(n):
        if abs(U[i, i]) < eps:
            print("LU: There is a zero pivot!")
            return np.zeros(1)
        if abs(U[i, i] - 1.0) < eps:
            L[i + 1:, i] = U[i + 1:, i]
        else:
            L[i + 1:, i] = U[i + 1:, i] / U[i, i]
        U[i + 1:, :] = U[i + 1:, :] - L[i + 1:, i] * U[i, :]

    E1 = np.mat(np.eye(n))  # 这个E1用来求U的逆
    for i in range(n - 1, -1, -1):
        if abs(U[i, i] - 1.0) > eps:
            # 对角元单位化
            E1[i, :] = E1[i, :] / U[i, i]
            # U[i, :] = U[i, :] / U[i, i]

        E1[:i, :] = E1[:i, :] - U[:i, i] * E1[i, :]
        # U[:i, :] = U[:i, :] - U[:i, i] * U[i, :]

    # 当然，我们还可以来求一下下三角阵L的逆
    E2 = np.mat(np.eye(n))
    for i in range(n):
        E2[i + 1:, :] = E2[i + 1:, :] - L[i + 1:, i] * E2[i, :]
        # L[i + 1:, :] = L[i + 1:, :] - L[i + 1:, i] * L[i, :]

    return np.array(E1 * E2)


if __name__ == "__main__":
    # A = np.mat(np.random.randn(3, 3))
    A = np.array([[1, 2, 3],  # for test
                  [3, 5, 2],
                  [1, 0, 1]])
    X = LU_inv(A)
    print(A @ X)
