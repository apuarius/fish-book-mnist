import numpy as np
#适用一维数组求梯度
def numerical_gradient_1d(f, x):
    """
       f:目标函数(Loss函数)
       x:参数(只能是一维数组)
       """
    #准备工作
    h = 1e-4    #步长
    grad = np.zeros_like(x) #初始化数组用来存储梯度数据
    #开始遍历每一个x
    for i in range(x.size):
        #1.存储当前的值
        tep_value = x[i]
        #2.向前步长的偏导
        x[i] = tep_value + h
        fxh1 = f(x)
        #3.向后步长的偏导
        x[i] = tep_value - h
        fxh2 = f(x)
        #4.求出梯度并保存
        grad[i] = (fxh1-fxh2)/(2*h)
        #5.还原
        x[i] = tep_value

    return grad
if __name__ == '__main__':
    # --- 测试代码 ---

    # 定义目标函数: f(x) = x0^2 + x1^2
    def function_2(x):
        return x[0] ** 2 + x[1] ** 2
        # 也可以写成: return np.sum(x**2)


    # 准备输入点: (3.0, 4.0)
    # 注意：必须用浮点数，不要写成整数 [3, 4]
    x = np.array([3.0, 4.0])

    # 计算梯度
    grad = numerical_gradient_1d(function_2, x)

    print("输入 x:", x)
    print("计算出的梯度:", grad)