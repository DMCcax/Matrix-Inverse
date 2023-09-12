import numpy as np
from same_seeds import same_seeds
np.set_printoptions(suppress=True, precision=4)
same_seeds(1)

eps = 1e-4  # 允许的最大误差

def Conjugate_gradient_Inv(m):
    # k_max = m.shape[1]
    k_max = 10
    A = np.mat(m)
    x = np.mat(np.zeros([A.shape[0], A.shape[1]]))

    if abs(A.T - A).sum() < eps:
        for j in range(A.shape[1]):
            xx = np.mat(np.zeros([A.shape[0], 1]))
            b = np.mat(np.zeros([A.shape[0], 1]))
            b[j] = 1
            g = A * xx - b
            g_last = g.copy()
            d = 0

            for i in range(k_max + 1):
                if abs(abs(g).sum()) < eps:
                    break
                if i == 0:
                    d = -g
                    beta = 0
                else:
                    beta = (g.T * (g - g_last) / (g_last.T * g_last)).item()
                    d = -g + beta * d
                dTAd = (d.T * A * d).item()
                if (dTAd<=0):
                    print("Conjugate_gradient: It's a Non-positive definite matrix !")
                    return np.zeros(1)
                alpha = (-d.T * g / dTAd).item()
                xx = xx + alpha * d
                g_last = g
                g = A * xx - b

            x[:, j] = xx

    else:
        B = A.T * A     # A.T * A * x = A.T * b
        for j in range(B.shape[1]):
            xx = np.mat(np.zeros([B.shape[0], 1]))
            b = A.T[:, j]
            g = B * xx - b
            g_last = g.copy()
            d = 0

            for i in range(k_max + 1):
                if abs(abs(g).sum()) < eps:
                    break
                if i == 0:
                    d = -g
                    beta = 0
                else:
                    beta = (g.T * (g - g_last) / (g_last.T * g_last)).item()
                    d = -g + beta * d
                dTBd = (d.T * B * d).item()
                if (dTBd <= 0):
                    print("Conjugate_gradient: It's a Non-positive definite matrix !")
                    return np.zeros(1)
                alpha = (-d.T * g / dTBd).item()
                xx = xx + alpha * d
                g_last = g
                g = B * xx - b

            x[:, j] = xx

    return np.array(x)


if __name__ == '__main__':
    A = np.array([[3, 1, 1],  # for test
              [1, 5, 2],
              [1, 2, 1]])
    # A = np.random.randn(3, 3)
    X = Conjugate_gradient_Inv(A)
    print(np.dot(A, X))