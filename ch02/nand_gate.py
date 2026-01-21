#与非门（输入均为1时输出0，否则输出1）
import numpy as np

def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b =0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
if __name__ == '__main__':
    print(NAND(0, 0))#1
    print(NAND(0, 1))#1
    print(NAND(1, 0))#1
    print(NAND(1, 1))#0