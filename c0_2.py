# 这部分开始利用pytorch手动搭建一个模型
import torch
batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10
x = torch.randn(batch_n, input_data)    #100*1000
y = torch.randn(batch_n,output_data)

w1 = torch.randn(input_data, hidden_layer) # 1000*100
w2 = torch.randn(hidden_layer,output_data)  # 100*10

epoch_n = 20
learning_rate = 1e-6

for epoch in range(epoch_n):
    h1 = x.mm(w1)  # (100*1000)*(1000*100)=(100*100)
    h1 = h1.clamp(min=0)  # 将小于0 的部分赋为0，效果类似于RELU
    y_pred = h1.mm(w2)  # (100*100)*(100*10)=(100*10)

    loss = (y_pred - y).pow(2).sum()
    print('Epoch:{},Loss:{:.4f}'.format(epoch, loss))

    grad_y_pred = 2 * (y_pred - y)
    grad_w2 = h1.t().mm(grad_y_pred)  # 100*10
    grad_h = grad_y_pred.clone()
    grad_h = grad_h.mm(w2.t())  # (100*10)*(10*100) = 100*100
    grad_h.clamp_(min=0)  # 将小于0 的部分赋为0，效果类似于RELU
    grad_w1 = x.t().mm(grad_h)  # (1000*100)*(100*100) =1000*100

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

import torch
from torch.autograd import Variable

batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10

x = Variable(torch.randn(batch_n, input_data), requires_grad=False)
y = Variable(torch.randn(batch_n, output_data), requires_grad=False)

w1 = Variable(torch.randn(input_data, hidden_layer), requires_grad=True)
w2 = Variable(torch.randn(hidden_layer, output_data), requires_grad=True)

epoch_n = 20
learning_rate = 1e-6

for epoch in range(epoch_n):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print('Epoch:{},Loss:{:.4f}'.format(epoch, loss.item()))

    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    w1.grad.data.zero_()  # 将得到的各个参数的梯度值通过grad.data.zero_()全部置零，如果不置零，则被一直累加
    w2.grad.data.zero_()
print("***************************************************************************************************")
import torch
from torch.autograd import Variable

batch_n = 64
hidden_layer = 100
input_data = 1000
output_data = 10


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_d, w1, w2):
        x = torch.mm(input_d, w1)
        x = torch.clamp(x, min=0)
        x = torch.mm(x, w2)
        return x

    def backward(self):
        pass


model = Model()

x = Variable(torch.randn(batch_n, input_data), requires_grad=False)
y = Variable(torch.randn(batch_n, output_data), requires_grad=False)

w1 = Variable(torch.randn(input_data, hidden_layer), requires_grad=True)
w2 = Variable(torch.randn(hidden_layer, output_data), requires_grad=True)

epoch_n = 30
learning_rate = 1e-6

for epoch in range(epoch_n):
    y_pred = model(x, w1, w2)

    loss = (y_pred - y).pow(2).sum()
    print('Epoch:{},Loss:{:.4f}'.format(epoch, loss.item()))

    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()


