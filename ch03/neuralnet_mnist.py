import pickle
import sys, os
import numpy as np
#手写数字识别推理网络
#1.设置路径，找到dataset文件夹
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

#2.引入激活函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

#3.获取数据
def get_data():
#normalize = Ture:把像素值0~255 变成 0.0~1.0 归一化
#flatten = True:把图像变成一维数组
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True,one_hot_label=False)
    return x_test, t_test

#4.初始化网络
def init_network():
    #从pkl文件里读入训练好的权重
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

#5.推理核心
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = sigmoid(np.dot(x, W1) + b1)
    a2 = sigmoid(np.dot(a1, W2) + b2)
    y = softmax(np.dot(a2, W3) + b3)
    return y

#主程序
if __name__ == '__main__':
    x, t = get_data() #x代表图片 t代表标签
    network = init_network()

    batch_size = 100 #批处理
    accuracy_cnt = 0
    total_count = len(x)

    print("推理中")
    for i in range(0,total_count,batch_size):     #以100为步长生成序列
        x_batch = x[i:i + batch_size]               #从 x 中取出第 i 到 i+100 之间的数据

        # y_batch 的形状通常是 (100, 10)，表示100张图片，每张图片有10个类别的概率分数
        # 将这100张图片一次性放入网络
        y_batch = predict(network, x_batch)

        # axis=1 表示在第1维度（行方向/类别方向）上寻找最大值的索引
        # 结果 p 是一个包含100个数字的数组，代表这100张图片预测的数字
        p = np.argmax(y_batch,axis = 1)

        # 1. 生成 True/False 的对比数组
        match_array = (p == t[i:i + batch_size])

        # 2. 统计这个数组里有多少个 True，并加到总分里
        accuracy_cnt += np.sum(match_array)

    print("Accuracy:"+str(float(accuracy_cnt)/total_count))

