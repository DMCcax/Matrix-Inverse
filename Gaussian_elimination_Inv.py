import numpy as np
from same_seeds import same_seeds

np.set_printoptions(suppress=True, precision=5)
same_seeds(1)


def Gaussian_Inv(m):
    eps = 1e-3
    A = np.mat(m.copy(), dtype=float)
    E = np.mat(np.eye(A.shape[0]))
    dim = A.shape[0]
    for i in range(dim - 1):
        if abs(A[i, i]) < eps:
            print('Gaussian: There is a zero pivot!')
            return np.zeros(1)
        if abs(A[i, i] - 1) > eps:
            t = A[i, i]
            for k in range(dim):
                E[i, k] /= t
                A[i, k] /= t
        for j in range(i+1, dim):
            t = A[j, i]
            for k in range(dim):
                E[j, k] -= E[i, k] * t
                A[j, k] -= A[i, k] * t
    if abs(A[-1, -1]) < eps:
        print('Gaussian: There is a zero pivot!')
        return np.zeros(1)
    else:
        if abs(A[-1, -1] - 1) > eps:
            for k in range(dim):
                E[-1, k] /= A[-1, -1]
            A[-1, -1] /= A[-1, -1]

    for i in range(dim - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            t = A[j, i]
            for k in range(dim):
                E[j, k] -= E[i, k] * t
                A[j, k] -= A[i, k] * t

    return np.array(E)


if __name__ == '__main__':
    A = np.array([[1, 2, 3],  # for test
                  [3, 5, 2],
                  [1, 0, 1]])
    # A = np.random.randn(3, 3)
    X = Gaussian_Inv(A)
    print(A @ X)
