import numpy as np
from same_seeds import same_seeds

same_seeds(2)
np.set_printoptions(suppress=True, precision=5)


def LUP_inv(m):
    eps = 1e-3
    n = m.shape[0]
    L = np.mat(np.eye(n), dtype=float)
    P = np.mat(np.eye(n), dtype=float)
    U = np.mat(m.copy())
    for i in range(n):
        swap_line = np.argmax(abs(U[i:, i])) + i
        if swap_line != i:
            P[[i, swap_line]] = P[[swap_line, i]]
            U[[i, swap_line]] = U[[swap_line, i]]
            if i > 0:
                L[[i, swap_line], :i] = L[[swap_line, i], :i]

        if abs(U[i, i]) < eps:
            print("LUP: There is no non-zero pivot!")
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

    return np.array(E1 * E2 * P)


if __name__ == "__main__":
    # A = np.mat(np.random.randn(4, 4))
    A = np.array([[1, 2, 3, 4],  # for test
                        [2, 3, 1, 5],
                        [3, 1, 2, 6],
                        [1, 3, 2, 6]], dtype=float)
    X = LUP_inv(A)
    print(A @ X)
