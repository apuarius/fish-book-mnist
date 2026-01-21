#或门（有一个1，输出1）
import numpy as np

def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b =-0.3
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
if __name__ == '__main__':
    print(OR(0, 0))#0
    print(OR(0, 1))#1
    print(OR(1, 0))#1
    print(OR(1, 1))#1