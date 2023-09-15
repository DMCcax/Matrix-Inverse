import numpy as np
from same_seeds import same_seeds
from scipy import linalg

np.set_printoptions(suppress=True, precision=5)
same_seeds(0)


def Cholesky(A):
    L = np.mat(np.zeros([A.shape[0], A.shape[1]]), dtype=float)
    if A.shape[0] == 1:
        return np.sqrt(A[0, 0])
    else:
        L[0, 0] = np.sqrt(A[0, 0])
        for i in range(1, A.shape[0]):
            L[i, 0] = A[i, 0] / L[0, 0]
        L[1:, 1:] = Cholesky(A[1:, 1:] - L[1:, 0] * L[1:, 0].T)
        return L

def Cholesky_Inv(m):
    eps = 1e-3
    dim = m.shape[0]
    A = np.mat(m.copy())
    L = Cholesky(A)

    E = np.mat(np.eye(dim))
    for i in range(dim):
        if abs(L[i, i] - 1.0) > eps:
            for k in range(0, i+1):
                # 对角元单位化
                E[i, k] = E[i, k] / L[i, i]
                # U[i, :] = U[i, :] / U[i, i]
        for j in range(i+1, dim):
            for k in range(0, j+1):
                E[j, k] = E[j, k] - L[j, i] * E[i, k]
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

