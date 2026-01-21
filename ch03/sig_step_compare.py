import numpy as np
import matplotlib.pyplot as plt

#阶跃函数与sigmoid函数的对比
#1.阶跃函数（感知机用的）：只有0和1，突变
def step_function(x):
    return np.array(x > 0, dtype=int)
#2.Sigmoid函数（神经网络用的）：平滑的S型曲线
def sigmoid(x):
    return 1/(1+np.exp(-x))
if __name__ == '__main__':
#生成从-0.5到5.0的数据，步长0.1
    x = np.arange(-5.0,5.0,0.1)
    y1 = step_function(x)
    y2 = sigmoid(x)
    #开始绘图
    #阶跃函数用蓝色虚线表示，sigmoid函数用红色实线表示。
    plt.plot(x,y1,linestyle="--",color='blue',label='Step Function')
    plt.plot(x,y2,color='red',label='Sigmoid')
    plt.ylim(-0.1,1.1)#限制y轴的范围
    plt.title('Step Function vs Sigmoid')
    plt.legend()
    plt.show()

