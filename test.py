# 测试程序，测试各种方法的时间，并排序

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
epoch = 10
def test(M):
    Inv_correct = []    # 用于查看哪种结果计算错误，正数表示结果正确；负数表示结果错误
    Time_list = {}  # 各方法运行所用时间
    Fun_list = [Gaussian_Inv,
                LU_inv,
                LUP_inv,
                Cholesky_Inv,
                QR_inv]

    try:
        start = time.time()
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

    return Time_list, inv_result


if __name__ == "__main__":
    M = np.random.rand(80, 80)
    # M = np.eye(80, 80)
    M = M @ M.T
    # M = 1 * np.eye(80, dtype=float)

    # M = np.array([[3, 1, 1],  # for test
    #               [1, 5, 2],
    #               [1, 2, 1]])
    # M = M.T @ M

    Is_positive_matrix(M)

    T = {}  # 累加时间
    for i in range(epoch):
        print(f"\nepoch:[{i+1}/{epoch}]")
        Time_list, inv_result = test(M)
        for j in Time_list:
            if i == 0:
                T[j] = Time_list[j]
            else:
                T[j] = T[j] + Time_list[j]

        sort_result = sorted(T.items(), key=lambda x: x[1])
        for k in range(len(sort_result)):
            if inv_result.any():
                print(k + 1, ". ", sort_result[k][0], ":\t", 100 * sort_result[k][1] / T['inv'], "%")
            else:
                print(k + 1, ". ", sort_result[k][0], ":\t", 100 * sort_result[k][1])

