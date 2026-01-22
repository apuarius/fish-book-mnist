import numpy as np
#均方误差
def mean_squared_error(y, t):
    return 0.5*np.sum((y-t)**2)

#交叉熵误差
def cross_entropy_error(y, t):
    #如果是处理单张图，将一维数组转成二维矩阵，以实现后续的计算
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    #如果是处理多张图片，获取多张图片的数量
    batch_size = y.shape[0]
    #如果神经网络输出的元素个数与监督数据元素个数相同就是one-hot格式，将其转化为标签格式
    if y.size == t.size:
        t = t.argmax(axis=1)#获得最大值所在位置的索引
    #在y中找到监督数据正确对应位置的值的预测概率分布
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size

if __name__ == '__main__':
    #简单测试

    # 场景 1: 单张图片，预测得不错 (正确答案是索引 2)
    y_good = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    t_onehot = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    t_label = np.array([2])

    # 测试 A: One-hot 格式 (代码应该自动用 argmax 转换)
    loss_a = cross_entropy_error(y_good, t_onehot)
    print(f"测试 A (单张 One-hot): {loss_a:.4f} \t(预期: 约 0.51)")

    # 测试 B: 标签格式 (代码应该直接用)
    loss_b = cross_entropy_error(y_good, t_label)
    print(f"测试 B (单张 Label):   {loss_b:.4f} \t(预期: 约 0.51 -> 应该和A一样)")

    print("-" * 30)

    # 场景 2: Batch (2张图)，一张准一张不准
    # 图1: 准 (0.6概率) -> Loss约0.51
    # 图2: 烂 (0.1概率) -> Loss约2.30
    # 平均 Loss 应该是 (0.51 + 2.30) / 2 = 1.405
    y_batch = np.array([
        [0.1, 0.05, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 正确答案是2
        [0.1, 0.05, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0]  # 正确答案是0 (但预测给了5最高分)
    ])
    t_batch_label = np.array([2, 0])

    # 测试 C: Batch + 标签格式
    loss_c = cross_entropy_error(y_batch, t_batch_label)
    print(f"测试 C (Batch Label):  {loss_c:.4f} \t(预期: 约 1.405)")


