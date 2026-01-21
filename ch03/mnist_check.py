import sys, os
import numpy as np
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
#调用官方的minist.py文件下载数据集
# 第一次运行会从网上下载数据，可能需要几分钟，请耐心等待
print("正在加载/下载数据，请稍候...")
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# 打印一下数据的身材 (Shape)
print("\n--- 数据加载完成 ---")
print(f"训练图像 (x_train): {x_train.shape}") # (60000, 784)
print(f"训练标签 (t_train): {t_train.shape}") # (60000,)
print(f"测试图像 (x_test): {x_test.shape}")   # (10000, 784)
print(f"测试标签 (t_test): {t_test.shape}")   # (10000,)