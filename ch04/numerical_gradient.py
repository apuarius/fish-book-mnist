import numpy as np
#通数值微分求梯度

def numerical_gradient(f, x):
    """
    f:目标函数(Loss函数)
    x:参数(可以是任何形状的矩阵或张量)
    """
    h = 1e-4 #步长
    grad = np.zeros_like(x) #初始化数组用来存储梯度数据

    #创建迭代器
    #flags=['multi_index']获取当前位置索引，用于后续存储梯度数据
    #op_flags=['readwrite']开启读写权限，允许修改x的值
    it = np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])

    #只要没结束就继续
    while not it.finished:
        #1.获得坐标
        idx = it.multi_index

        #2.保存当前坐标的值
        idx_value = x[idx]

        #3.计算向前步长的偏导,浮点型防止结果被截断导致零步长变化
        x[idx] = float(idx_value) + h
        fh1 = f(x)

        #4.计算向后步长的偏导
        x[idx] = float(idx_value) - h
        fh2 = f(x)

        #5.求出梯度并保存
        grad[idx] = (fh1 - fh2) / (2*h)

        #6.还原
        x[idx] = idx_value

        #7.下一个
        it.iternext()

    return grad


if __name__ == '__main__':
    # 定义目标函数：求平方和
    def function_2(x):
        return np.sum(x ** 2)


    # 准备一个 2x3 的矩阵 (多维的)
    W = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

    print("--- 正在计算二维矩阵梯度 ---")
    print("输入 W:\n", W)

    # 调用函数
    grads = numerical_gradient(function_2, W)

    print("\n计算出的梯度 grads (应该是输入的2倍):\n", grads)
    #[[ 2.  4.  6.][ 8. 10. 12.]]




