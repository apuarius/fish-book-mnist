import numpy as np

def softmax(a):
    #找出最大值防止溢出
    c = np.max(a)
    #每个数都减去最大值再算指数
    exp = np.exp(a - c)
    #算出总和
    sum_exp = np.sum(exp)
    #算出概率（当前值/总和）
    y = exp / sum_exp
    return y

if __name__ == '__main__':
    a = np.array([0.3,2.9,4.0])
    y = softmax(a)
    print(f"原始分数：{a}")
    print(f"概率分布：{y}")
    print(f"概率之和：{np.sum(y)}")#应该等于1.0

