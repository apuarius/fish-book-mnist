#两层神经网络的实现
import numpy as np
import os,sys
sys.path.append(os.pardir)
from ch03.softmax import softmax
from ch03.sig_step_compare import sigmoid
from ch04.numerical_gradient import numerical_gradient
from ch04.test_loss import  cross_entropy_error

class two_layer_net:
    def __init__(self, input_size, hidden_size, output_size,weight_init_std=0.01):
        """
        初始化参数
        :param input_size: 输入层神经元数量
        :param hidden_size: 隐藏层神经元数量
        :param output_size: 输出层神经元数量
        :param weight_init_std: 设置随机初始化权重的范围
        """
        #初始化权重
        self.params = {}
        #第一层隐藏层
        #权重乘以缩放因子，让信号落在sigmoid函数的线性区防止梯度消失
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)

        #初始化隐藏层偏置
        self.params['b1'] = np.zeros(hidden_size)

        #第二层输出层
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        #初始化输出层偏置
        self.params['b2'] = np.zeros(output_size)

    def predict(self, X):
        #前向传播
        #解包
        W1,b1 = self.params['W1'],self.params['b1']
        W2,b2 = self.params['W2'],self.params['b2']

        #第一层
        a1 = np.dot(X,W1) + b1
        z1 = sigmoid(a1)

        #第二层
        a2 = np.dot(z1,W2) + b2
        y = a2

        return y

    def loss(self, X, t):
        #损失函数
        y = self.predict(X)

        y = softmax(y)

        loss = cross_entropy_error(y, t)

        return loss

    def accuracy(self, X, t):
        #计算识别精度
        y = self.predict(X)

        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(t == y) / float((X.shape[0]))
        return accuracy

    def numerical_gradient(self, X, t):
        loss_W = lambda _: self.loss(X, t)

        #数值微分计算梯度
        #用字典存储对应的梯度数据
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

# --- 测试代码 ---

if __name__ == '__main__':

    # 实例化网络

    net = two_layer_net(input_size=784, hidden_size=100, output_size=10)



    # 打印参数形状检查

    print("W1形状:", net.params['W1'].shape)

    print("b1形状:", net.params['b1'].shape)

    print("W2形状:", net.params['W2'].shape)

    print("b2形状:", net.params['b2'].shape)

    print("类定义成功，没有报错！")

















