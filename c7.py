# 循环神经网络
import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
# %matplotlib inline
# 数据预处理
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean = [0.5],std = [0.5])])
# 读取数据
dataset_train = datasets.MNIST(root = './data',
                              transform = transform,
                              train = True,
                              download = False)
dataset_test = datasets.MNIST(root = './data',
                             transform = transform,
                             train = False)
# 加载数据
train_load = torch.utils.data.DataLoader(dataset = dataset_train,
                                        batch_size = 64,
                                        shuffle = True)
test_load = torch.utils.data.DataLoader(dataset = dataset_test,
                                       batch_size = 64,
                                       shuffle = True)
# 数据可视化
images, label = next(iter(train_load))

images_example = torchvision.utils.make_grid(images)
images_example = images_example.numpy().transpose(1,2,0) # 将图像的通道值置换到最后的维度，符合图像的格式
mean = [0.5,0.5,0.5]
std = [0.5,0.5,0.5]
images_example = images_example * std + mean
plt.imshow(images_example)
plt.show()


# 搭建RNN网络
class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(
            input_size=28,
            hidden_size=128,
            num_layers=2,
            batch_first=True)
        self.output = torch.nn.Linear(128, 10)

    def forward(self, input):
        output, _ = self.rnn(input, None)
        output = self.output(output[:, -1, :])  # 去掉中间的batch_size维度，使得输出为图像的维度
        return output


model = RNN()
print(model)
# 设置优化方法和损失函数
optimizer = torch.optim.Adam(model.parameters())
loss_f = torch.nn.CrossEntropyLoss()
# 训练函数
epoch_n = 10
for epoch in range(epoch_n):
    running_loss = 0.0
    running_correct = 0.0
    testing_correct = 0.0
    print('Epoch{}/{}'.format(epoch, epoch_n))
    print('-' * 10)

    for data in train_load:
        X_train, Y_train = data
        X_train = X_train.view(-1, 28,
                               28)  # 这里-1表示一个不确定的数，就是你如果不确定你想要batch size多大，但是你很肯定要reshape成28步，每步28维，那不确定的地方就可以写成-1
        X_train, Y_train = Variable(X_train), Variable(Y_train)
        y_pred = model(X_train)
        loss = loss_f(y_pred, Y_train)
        _, pred = torch.max(y_pred.data, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()
        running_correct += torch.sum(pred == Y_train.data)

    for data in test_load:
        X_test, Y_test = data
        X_test = X_test.view(-1, 28, 28)
        X_test, Y_test = Variable(X_test), Variable(Y_test)
        output = model(X_test)
        _, pred = torch.max(output.data, 1)
        testing_correct += torch.sum(pred == Y_test.data)

print('Loss is:{:.4f}, Train ACC is:{:.4f}%, Test ACC is:{:.4f}'.format(running_loss / len(dataset_train),
                                                                        100 * running_correct / len(dataset_train),
                                                                        100 * testing_correct / len(dataset_test)))
