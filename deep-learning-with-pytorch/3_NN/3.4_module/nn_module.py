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
class module_net(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(module_net, self).__init__()
        self.layer1 = nn.Linear(num_input, num_hidden)
        self.layer2 = nn.Tanh()
        self.layer3 = nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

mo_net = module_net(2, 4, 1)

# 访问模型中的某层可以直接通过名字
print(mo_net.layer1)
print(mo_net.layer1.weight)

# 定义优化器
optim = torch.optim.SGD(mo_net.parameters(), 1.)

# 定义损失函数
# BCEWithLogitsLoss包含sigmoid函数
criterion = nn.BCEWithLogitsLoss()

# 我们训练 10000 次
for e in range(10000):
    out = mo_net(Variable(x))
    loss = criterion(out, Variable(y))
    optim.zero_grad()
    loss.backward()
    optim.step()
    if (e + 1) % 1000 == 0:
        print('epoch: {}, loss: {}'.format(e+1, loss.data))


# ----------------------------------------------------------------------------------------------------------------------
# 保存模型
torch.save(mo_net.state_dict(), 'save_module_net.pth')


