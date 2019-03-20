import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# 准备数据
torch.manual_seed(2017)

# 读入数据 x 和 y
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

plt.plot(x_train, y_train, 'bo')
plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# 构建线性回归模型
def linear_model(x):
    return x * w + b


# 构建损失函数
def get_loss(y_, y):
    return torch.mean((y_ - y_train) ** 2)


# ----------------------------------------------------------------------------------------------------------------------
# 模型初始化
# 转换成 Tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

x_train = Variable(x_train)
y_train = Variable(y_train)

# 定义参数 w 和 b
w = Variable(torch.randn(1), requires_grad=True) # 随机初始化
b = Variable(torch.zeros(1), requires_grad=True) # 使用 0 进行初始化

y_ = linear_model(x_train)


plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# 训练一次
loss = get_loss(y_, y_train)

# 自动求导
loss.backward()

# 更新一次参数
w.data = w.data - 1e-2 * w.grad.data
b.data = b.data - 1e-2 * b.grad.data

y_ = linear_model(x_train)

plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# 训练多次
for e in range(10):  # 进行 10 次更新
    y_ = linear_model(x_train)
    loss = get_loss(y_, y_train)

    if w.grad is not None:
        w.grad.zero_()  # 记得归零梯度
    if b.grad is not None:
        b.grad.zero_()  # 记得归零梯度
    loss.backward()

    w.data = w.data - 1e-2 * w.grad.data  # 更新 w
    b.data = b.data - 1e-2 * b.grad.data  # 更新 b
    print('epoch: {}, loss: {}'.format(e, loss.data))

y_ = linear_model(x_train)

plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.show()

