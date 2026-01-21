#异或门（仅当一方为1时输出1，否则输出0）
import numpy as np
from and_gate import AND
from or_gate import  OR
from nand_gate import NAND

def XOR(x1,x2):
    S1 = NAND(x1,x2)  #与非门的输出
    S2 = OR(x1,x2)    #或门的输出
    y = AND(S1,S2)    #将与非门的输出和或门的输出作为与与门的输入最后组成异或门
    return y
if __name__ == '__main__':
    print(XOR(0, 0))#0
    print(XOR(0, 1))#1
    print(XOR(1, 0))#1
    print(XOR(1, 1))#0

