# Matrix-Inverse
Several ways to find the inverse of a matrix

这个项目旨在不调用数值计算库的前提下，寻找求中等规模矩阵的逆的简便方法；

参与比较的方法有：高斯消元法、LU、LUP分解、QR分解以及Cholesky分解；

"test.py"文件用于测试以上方法的时间并排序，比较的基准为numpy的求逆方法。


在测试矩阵为80*80的对称矩阵，运行1000次求平均值的条件下，Cholesky最快，为基准的9%左右，其次是LU（11%）、LUP（13%）、QR（27%），最后是高斯消元法（2到3倍）。



各个方法的限制：

1.高斯、LU：要求主元不为0；

2.LUP、QR：只要逆矩阵存在即可求；

3.矩阵所有对角线元素为正才可以用Cholesky分解。
