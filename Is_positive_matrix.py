# 判断是否对称、正定

import numpy as np

def Is_positive_matrix(m):
    dim = m.shape[0]
    A = np.mat(m, dtype=float)
    p = 1
    if abs(m - m.T).sum() > 1e-4:
        print("Asymmetric matrix")
        p = -1
        return p

    for i in range(dim):
        if m[i, i] < 0:
            p = -1
            break
        else:
            sum = 0
            for j in range(dim):
                sum = sum + abs(m[i, j])
            if 2 * abs(m[i, i]) <= sum:
                p = 0

    if p == 1:
        print('Positive')
    elif p == 0:
        print('May Non-positive')
    else:
        print('Non-positive')
    return p

if __name__ == '__main__':
    A = np.array([[3, 1, 1],  # for test
                [1, 5, 2],
                [1, 2, 1]])
    # A = np.random.randn(3, 3)
    Is_positive_matrix(A)
