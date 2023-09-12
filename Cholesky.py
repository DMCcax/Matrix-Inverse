import numpy as np
from same_seeds import same_seeds
from scipy import linalg

np.set_printoptions(suppress=True, precision=5)
same_seeds(0)


def Cholesky(A):
    L = np.mat(A, dtype=float)
    if A.shape[0] == 1:
        return np.sqrt(A[0, 0])
    else:
        L[0, 0] = np.sqrt(A[0, 0])
        L[1:, 0] = A[1:, 0] / L[0, 0]
        L[1:, 1:] = Cholesky(A[1:, 1:] - L[1:, 0] * L[1:, 0].T)
        return L

def Cholesky_Inv(m):
    eps = 1e-3
    n = m.shape[0]
    A = np.mat(m.copy())
    L = Cholesky(A)

    E = np.mat(np.eye(n))
    for i in range(n):
        if abs(L[i, i] - 1.0) > eps:
            # 对角元单位化
            E[i, :] = E[i, :] / L[i, i]
            # L[i, :] = L[i, :] / L[i, i]

        E[i + 1:, :] = E[i + 1:, :] - L[i + 1:, i] * E[i, :]
        # L[i + 1:, :] = L[i + 1:, :] - L[i + 1:, i] * L[i, :]
    return np.array(E.T * E)



if __name__ == '__main__':
    A = np.random.randn(4, 4)
    A = A @ A.T
    # A = np.array([[4, 2, 1],  # for test
    #               [2, 3, 0],
    #               [1, 0, 2]])
    X = Cholesky_Inv(A)

    print(X @ A)

