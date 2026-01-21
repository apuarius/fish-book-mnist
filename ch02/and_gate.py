#与门(输入均为1输出1，否则输出0)
import numpy as np

def AND(x1,x2):
    x = np.array([x1,x2])   #输入
    w = np.array([0.5,0.5]) #权重
    b = -0.7 #偏置
    tmp = np.sum(w*x) + b #感知机公式 y = w*x + b
    if tmp <= 0:     #小于等于0返回0否则返回1
        return 0
    else:
        return 1
if __name__ == '__main__':
    print(AND(0,0))#0
    print(AND(0,1))#0
    print(AND(1,0))#0
    print(AND(1,1))#1