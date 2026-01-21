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

    accuracy_cnt = 0
    total_count = len(x)

    print("推理中")
    for i in range(total_count):
         y = predict(network, x[i]) #y是一个包含10个概率的数组
         p = np.argmax(y) #获取概率最大的那个索引
         if p == t[i]: #如果预测结果的索引等于正确答案的索引
             accuracy_cnt += 1

    print("Accuracy:"+str(float(accuracy_cnt)/total_count))

