import numpy as np

#1.简单的多维数组
A = np.array([[1, 2, 3],[4, 5, 6]])
print(f"A:\n{A}")
print(f"维数（ndim）: {A.ndim}")#2维
print(f"形状（shape）: {A.shape}")#2行3列

#2.矩阵乘法
B = np.array([[1, 2],[3, 4],[5, 6]])
print(f"B:\n{B}")
print(f"维数（ndim）: {B.ndim}")#2维
print(f"B.shape: {B.shape}")#3行2列

#A是（2，3），B是（3，2）
#中间的‘3’和‘3’对上了，可以乘
Y = np.dot(A,B)
print(f"Y:\n{Y}")
print(f"Y.shape: {Y.shape}")#（2，2）