# 实现手写数字集
import torch
import numpy
import torchvision
import matplotlib.pyplot as plt
# %matplotlib inline
from torchvision import datasets, transforms   # torchvision包的主要功能是实现数据的处理、导入和预览等
from torch.autograd import Variable
#  transform用于指定导入数据集时需要对数据进行哪种变换操作，在后面会介绍详细的变换操作类型，注意，要提前定义这些变换操作
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                transforms.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5])])
data_train = datasets.MNIST(root = './data/',
                           transform = transform,
                           train = True,
                           download = False)
data_test = datasets.MNIST(root = './data/',
                          transform = transform,
                          train = False)
data_loader_train = torch.utils.data.DataLoader(dataset = data_train, batch_size = 64, shuffle = True)
data_loader_test = torch.utils.data.DataLoader(dataset = data_test, batch_size = 64, shuffle = True)

# 装载完成后，我们可以选取其中一个批次的数据进行预览。进行数据预览的代码如下：
images, labels = next(iter(data_loader_train))
img = torchvision.utils.make_grid(images)
# 这里的图像数据的size是（channel,height,weight）,而我们显示图像的size顺序为（height,weight,channel）

img = img.numpy().transpose(1,2,0)          # 转为（height,weight,channel）
std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img * std + mean                     # 将上面Normalize后的图像还原
print([labels[i].item() for i in range(64)])  # labels是tensor数据，要显示他的值要用.item()
plt.imshow(img)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)  # 对得到的多层的参数进行扁平化处理，使之能与全连接层连接
        x = self.dense(x)
        return x


model = Model()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

print(model)

epoch_n = 5
for epoch in range(epoch_n):
    running_loss = 0.0
    running_correct = 0
    print('Epoch {}/{}'.format(epoch, epoch_n))
    print('-' * 10)
    for data in data_loader_train:
        X_train, Y_train = data
        X_train, Y_train = Variable(X_train), Variable(Y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)  # 注意：这句的意思是：返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
        optimizer.zero_grad()
        loss = cost(outputs, Y_train)

        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()
        running_correct += torch.sum(pred == Y_train.data)

    testing_correct = 0
    for data in data_loader_test:
        X_test, Y_test = data
        X_test, Y_test = Variable(X_test), Variable(Y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == Y_test.data)
    print('Loss is :{:.4f},Train Accuracy is : {:.4f}%, Test Accuracy is :{:.4f}'.format(running_loss / len(data_train),
                                                                                         100 * running_correct / len(
                                                                                             data_train),
                                                                                         100 * testing_correct / len(
                                                                                             data_test)))




data_loader_test = torch.utils.data.DataLoader(dataset = data_test, batch_size = 4, shuffle = True)

X_test, Y_test = next(iter(data_loader_test))
inputs = Variable(X_test)
pred = model(inputs)
_, pred = torch.max(pred, 1)    # 注意：这句的意思是：返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）

print('Predict Label is :', [i.item() for i in pred.data])    # i 是tensor类型，要显示数值要用.item()
print('Real Label is :', [i.item() for i in Y_test])

img = torchvision.utils.make_grid(X_test)
img = img.numpy().transpose(1,2,0)

std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img * std + mean
plt.imshow(img)

