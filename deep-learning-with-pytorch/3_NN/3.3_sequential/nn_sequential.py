import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# 数据整理
np.random.seed(1)
m = 400 # 样本数量
N = int(m/2) # 每一类的点的个数
D = 2 # 维度
x = np.zeros((m, D))
y = np.zeros((m, 1), dtype='uint8') # label 向量，0 表示红色，1 表示蓝色
a = 4

for j in range(2):
    ix = range(N*j,N*(j+1))
    t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
    r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
    x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

# plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)
# plt.show()

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

# ----------------------------------------------------------------------------------------------------------------------
# 模型训练

# Sequential
seq_net = nn.Sequential(
    nn.Linear(2, 4), # PyTorch 中的线性层，wx + b
    nn.Tanh(),
    nn.Linear(4, 1)
)

# 序列模块可以通过索引访问每一层
# print(seq_net[0])
# print(seq_net[0].weight)

# 通过 parameters 可以取得模型的参数
param = seq_net.parameters()

# 定义优化器
optim = torch.optim.SGD(param, 1.)
# 定义损失函数
criterion = nn.BCEWithLogitsLoss()

# 我们训练 10000 次
for e in range(10000):
    out = seq_net(Variable(x))
    loss = criterion(out, Variable(y))
    optim.zero_grad()
    loss.backward()
    optim.step()
    if (e + 1) % 1000 == 0:
        print('epoch: {}, loss: {}'.format(e+1, loss.data))


# ----------------------------------------------------------------------------------------------------------------------
# 模型可视化
def plot_decision_boundary(model, x, y):
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)


def plot_seq(x):
    out = F.sigmoid(seq_net(Variable(torch.from_numpy(x).float()))).data.numpy()
    out = (out > 0.5) * 1
    return out


# plot_decision_boundary(lambda x: plot_seq(x), x.numpy(), y.numpy())
# plt.title('sequential')
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# 保存和加载模型

print(seq_net)
print(seq_net[0].weight)

# 方法一、 保存模型结构和参数
# 将参数和模型保存在一起
torch.save(seq_net, 'save_seq_net.pth')
# 读取保存的模型
seq_net1 = torch.load('save_seq_net.pth')
print(seq_net1)
print(seq_net1[0].weight)

# 方法二、 保存参数
# 保存模型参数
torch.save(seq_net.state_dict(), 'save_seq_net_params.pth')
seq_net2 = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)
seq_net2.load_state_dict(torch.load('save_seq_net_params.pth'))
print(seq_net2)
print(seq_net2[0].weight)



