# 用于比较这些方法的快慢以及正确性

import numpy as np
import time
from numpy.linalg import inv
from Is_positive_matrix import Is_positive_matrix
from Gaussian_elimination_Inv import Gaussian_Inv
from LU import LU_inv
from LUP import LUP_inv
from Cholesky import Cholesky_Inv
from QR import QR_inv
from same_seeds import same_seeds

same_seeds(0)
np.set_printoptions(suppress=True, precision=4)

eps = 1e-3
epoch = 1

M = np.random.randn(80, 80)
# M = np.eye(80, 80)
M = M @ M.T     # 将矩阵转为对称阵
# M = 1 * np.eye(80, dtype=float)

# M = np.array([[3, 1, 1],  # for test
#               [1, 5, 2],
#               [1, 2, 1]])
# M = M.T @ M

Inv_correct = []  # 用于查看哪种结果计算错误，正数表示结果正确；负数表示结果错误
Time_list = {}  # 各方法运行所用时间
Fun_list = [Gaussian_Inv,
            LU_inv,
            LUP_inv,
            Cholesky_Inv,
            QR_inv]

Is_positive_matrix(M)

try:
    start = time.time()
    for j in range(epoch):
        inv_result = inv(M)
    total_time = time.time() - start
    Time_list[inv.__name__] = total_time
except np.linalg.LinAlgError:
    print("This is a Singular Matrix!")
    inv_result = np.zeros_like(M)

k = 1
for i in Fun_list:
    try:
        start = time.time()
        for j in range(epoch):
            inv_result_1 = i(M)
        total_time = time.time() - start

        if inv_result_1.any():
            Time_list[i.__name__] = total_time
            print(i.__name__, "Done.")

            if abs(inv_result - inv_result_1).sum() < eps:
                Inv_correct.append(k)
            else:
                Inv_correct.append(-k)
        else:
            print(i.__name__, "doesn't work!\n")
            Inv_correct.append(-k)

    except np.linalg.LinAlgError:
        print("This is a Singular Matrix!")
        print(i.__name__, "doesn't work!\n")
        continue

    k += 1

sort_result = sorted(Time_list.items(), key=lambda x: x[1])
for i in range(len(sort_result)):
    if inv_result.any():
        print(i + 1, ". ", sort_result[i][0], ":\t", 100 * sort_result[i][1] / Time_list['inv'], "%")
    else:
        print(i + 1, ". ", sort_result[i][0], ":\t", 100 * sort_result[i][1])

# Inv = np.array(Inv)
# print("Result of Inverse:\n", Inv[0])
