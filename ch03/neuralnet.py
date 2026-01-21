#简单的神经网络搭建
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def identity_function(x):
    return x
#初始化网络
def init_network():
    network = {}
    #第一层：输入(2)-隐藏(3)
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])#2行3列
    network['b1'] = np.array([0.1,0.2,0.3])
    #第二层：隐藏(3)-输出(2)
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])#3行2列
    network['b2'] = np.array([0.1,0.2,])
    #第三层：输出层
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])#2行2列
    network['b3'] = np.array([0.1,0.2])
    return network
def forward(network, x):
    #取出参数
    w1,w2,w3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    #第一层计算
    z1 = sigmoid(np.dot(x,w1) + b1)
    #第二层计算
    z2 = sigmoid(np.dot(z1,w2) + b2)
    #第三层计算
    y = identity_function(np.dot(z2,w3) + b3)
    return y
if __name__ == '__main__':
    network = init_network()
    x = np.array([[1.0,0.5]])#输入值
    y = forward(network,x)
    print(y)



