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
    def __init__(self,
                 num_input,
                 num_hidden1,
                 num_hidden2,
                 num_hidden3,
                 num_output):
        super(module_net, self).__init__()
        self.layer1 = nn.Linear(num_input, num_hidden1)
        self.layer1_a = nn.Tanh()
        self.layer2 = nn.Linear(num_hidden1, num_hidden2)
        self.layer2_a = nn.Tanh()
        self.layer3 = nn.Linear(num_hidden2, num_hidden3)
        self.layer3_a = nn.Tanh()
        self.layer4 = nn.Linear(num_hidden3, num_output)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer1_a(x)
        x = self.layer2(x)
        x = self.layer2_a(x)
        x = self.layer3(x)
        x = self.layer3_a(x)
        x = self.layer4(x)
        return x

net = module_net(2, 10, 10, 10, 1)

# 访问模型中的某层可以直接通过名字
print(net.layer1)
print(net.layer1.weight)

# 定义优化器
optim = torch.optim.SGD(net.parameters(), 0.1)

# 定义损失函数
# BCEWithLogitsLoss包含sigmoid函数
criterion = nn.BCEWithLogitsLoss()

# 我们训练 20000 次
for e in range(20000):
    out = net(Variable(x))
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


def plot_net(x):
    out = F.sigmoid(net(Variable(torch.from_numpy(x).float()))).data.numpy()
    out = (out > 0.5) * 1
    return out


plot_decision_boundary(lambda x: plot_net(x), x.numpy(), y.numpy())
plt.title('module')
plt.show()


