#数值微分训练手写数字识别
import numpy as np
import os,sys
import matplotlib.pyplot as plt
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from ch04.two_layer_net import two_layer_net

#读取数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
#列表存储批损失函数的数据
train_loss_list = []
train_acc_list = [] #记录训练集精度
test_acc_list = [] #记录测试集精度，用于检测是否过拟合
#超参数设置
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
#计算1个epoch需要循环多少次
iter_per_epoch = max(train_size / batch_size, 1)
#实例化网络
net = two_layer_net(input_size=784, hidden_size=50, output_size=10)
#训练循环
for i in range(iters_num):
    #获取随机批次索引
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #计算梯度
    grad = net.numerical_gradient(x_batch, t_batch)

    #使用循环更新字典参数
    for key in net.params:
        net.params[key] -= learning_rate * grad[key]

    #记录loss
    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #打印进度
    if i % iter_per_epoch == 0:
       train_acc =  net.accuracy(x_batch, t_batch)
       test_acc = net.accuracy(x_test, t_test)
       train_acc_list.append(train_acc)
       test_acc_list.append(test_acc)
       print("train acc,test acc | " + str(train_acc) + "," + str(test_acc))


    #画图
markers = {'train': 'o', 'test': 's'}

x = np.arange(len(train_acc_list))

plt.plot(x, train_acc_list, label='train acc')

plt.plot(x, test_acc_list, label='test acc', linestyle='--')

plt.xlabel("epochs")

plt.ylabel("accuracy")

plt.ylim(0, 1.0)

plt.legend(loc='lower right')

plt.show()

#数值微分速度级慢











