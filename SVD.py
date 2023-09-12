import numpy as np


def SVD_Inv(A):
    m, n = A.shape
    if m > n:
        sigma, V = np.linalg.eig(A.T @ A)
        # 将sigma 和V 按照特征值从大到小排列
        arg_sort = np.argsort(sigma)[::-1]
        sigma = np.sort(sigma)[::-1]
        V = V[:, arg_sort]

        # 对sigma进行平方根处理
        sigma_matrix = np.diag(np.sqrt(sigma))

        sigma_inv = np.linalg.inv(sigma_matrix)

        U = A @ V.T @ sigma_inv
        U = np.pad(U, pad_width=((0, 0), (0, m - n)))
        sigma_matrix = np.pad(sigma_matrix, pad_width=((0, m - n), (0, 0)))

    else:
        # 同m>n 只不过换成从U开始计算
        sigma, U = np.linalg.eig(A @ A.T)
        arg_sort = np.argsort(sigma)[::-1]
        sigma = np.sort(sigma)[::-1]
        U = U[:, arg_sort]

        sigma_matrix = np.diag(np.sqrt(sigma))
        sigma_inv = np.linalg.inv(sigma_matrix)
        V = sigma_inv @ U.T @ A
        V = np.pad(V, pad_width=((0, n - m), (0, 0)))

        sigma_matrix = np.pad(sigma_matrix, pad_width=((0, 0), (0, n - m)))

    sigma_matrix = np.mat(sigma_matrix)
    return np.array(V.T @ sigma_matrix.I @ U.T)


if __name__ == "__main__":
    a = np.random.randn(3,3)
    x = SVD_Inv(a)
    print(x @ a)
