#simple_net.py: 实现一个简单的神经网络类，用于练习数值微分求梯度
import numpy as np
import os,sys
sys.path.append(os.pardir)
from ch04.numerical_gradient import numerical_gradient
from ch04.test_loss import cross_entropy_error
from ch03.softmax import softmax

class simple_net:
    def __init__(self):
        #固定随机种子
        np.random.seed(0)
        #创建2*3的初始权重
        self.W1 = np.random.randn(2,3)

    def predict(self, X): #定义前向传播
        return np.dot(X, self.W1)

    def loss(self, X, t): #定义损失函数
       z = self.predict(X) #先进行前向传播计算
       y = softmax(z) #通过softmax函数转换成概率
       loss = cross_entropy_error(y, t)#通过交叉熵函数计算损失
       return loss

if __name__ == '__main__':
    #简单测试
    net = simple_net() #实例化对象
    #初始化数据和正确标签
    X = np.array([0.6, 0.9])
    t = np.array([0, 0 ,1])

    #构造伪参数
    f = lambda _: net.loss(X, t)
    #计算梯度
    grad = numerical_gradient(f, net.W1)
    print(f"计算出的梯度:\n{grad}")
    #计算出的梯度:[[ 0.44452826  0.14014461 -0.58467287][ 0.66679239  0.21021692 -0.87700931]]