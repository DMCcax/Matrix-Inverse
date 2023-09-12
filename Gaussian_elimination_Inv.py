import numpy as np
from same_seeds import same_seeds

np.set_printoptions(suppress=True, precision=5)
same_seeds(1)


def Gaussian_Inv(m):
    eps = 1e-3
    A = np.mat(m.copy(), dtype=float)
    B = np.mat(np.eye(A.shape[0]))
    n = A.shape[0]
    for k in range(n - 1):
        if abs(A[k, k]) < eps:
            print('Gaussian: There is a zero pivot!')
            return np.zeros(1)
        if abs(A[k, k] - 1) > eps:
            B[k, :] /= A[k, k]
            A[k, :] /= A[k, k]
        for i in range(k + 1, n):
            B[i, :] -= B[k, :] * A[i, k]
            A[i, :] -= A[k, :] * A[i, k]
    if abs(A[n - 1, n - 1]) < eps:
        print('Gaussian: There is a zero pivot!')
        return np.zeros(1)
    if abs(A[-1, -1]) > eps:
        B[-1, :] /= A[-1, -1]
        A[-1, :] /= A[-1, -1]

    for k in range(n - 1, 0, -1):
        for i in range(k - 1, -1, -1):
            B[i, :] -= B[k, :] * A[i, k]
            A[i, :] -= A[k, :] * A[i, k]

    return np.array(B)


if __name__ == '__main__':
    A = np.array([[1, 2, 3],  # for test
                  [3, 5, 2],
                  [1, 0, 1]])
    # A = np.random.randn(3, 3)
    X = Gaussian_Inv(A)
    print(A @ X)
