import numpy as np
import matplotlib.pyplot as plt

#ReLU函数：简单的“截断”逻辑
def relu(x):
    return np.maximum(0, x) #输入大于0输出x，输入小于0输出0

if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1) #生成从-0.5到5.0的数据，步长0.1
    y = relu(x)
    plt.plot(x, y,label='relu')
    plt.ylim(-0.1, 5.5) #调整视野
    plt.title('ReLU Function')
    plt.legend()
    plt.grid(True)#加一个网格背景
    plt.show()