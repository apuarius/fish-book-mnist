import numpy as np
import sys,os

#梯度下降法
sys.path.append(os.pardir)
from ch04.numerical_gradient import numerical_gradient

def gradient_descent(f,init_x,lr=0.01,step_num=100):
    """
    :param f:目标函数
    :param init_x: 初始位置
    :param lr: 学习率
    :param step_num:迭代次数
    """
    #1.复制一份数据，防止修改原始init_x
    x = init_x.copy()

    #2.开始循环
    for i in range(step_num):
        #调用梯度函数
        grad = numerical_gradient(f,x)
        #更新参数
        x -= lr * grad

    return x

# --- 测试代码 ---
if __name__ == '__main__':
    # 定义函数 f(x) = x0^2 + x1^2
    def function_2(x):
        return np.sum(x**2)

    # 初始位置 (-3.0, 4.0)
    init_x = np.array([-3.0, 4.0])

    # 开始下山！
    # 学习率设为 0.1，走 100 步
    final_x = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)

    print("初始位置:", init_x)
    print("最终位置:", final_x)
    #初始位置: [-3.  4.]
    #最终位置: [-6.11110793e-10  8.14814391e-10]
