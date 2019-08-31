# 这部分是利用pytorch进行迁移学习，迁移的是VGG16
import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import os            # os包集成了一些对文件路径和目录进行操作的类
import matplotlib.pyplot as plt
import time
# 读取数据
data_dir = 'DogsVSCats'
data_transform = {x:transforms.Compose([transforms.Scale([224, 224]),
                                       transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
                                        ]) for x in ['train', 'valid']}   # 这一步类似预处理,相比于手动搭建模型的方法，这里增加了预处理，图像大小为vgg的标准输入
image_datasets = {x:datasets.ImageFolder(root = os.path.join(data_dir,x),
                                        transform = data_transform[x]) for x in ['train', 'valid']}  # 这一步相当于读取数据
dataloader = {x:torch.utils.data.DataLoader(dataset = image_datasets[x],
                                           batch_size = 16,
                                           shuffle = True) for x in ['train', 'valid']}   # 读取完数据后，对数据进行装载
model = models.vgg16(pretrained=True)   # 调用Vgg16的预训练模型参数
print(model)
for parma in model.parameters():
    parma.requires_grad = False     # 冻结全部的梯度，使之梯度不进行更新

# 重新定义全连接层为我们想要的输出维度，进行二分类， 这里第一个全连接层的输入不应该对应是25088么？25088会报错？？修改后2048可以运行
# 这是因为开始的时候把图像裁剪到64*64导致的，标准vgg的输入是224*224的
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p = 0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p = 0.5),
                                       torch.nn.Linear(4096, 2))

print(model)  # 打印改变后的模型进行对比
#  定义好模型的损失函数和对参数进行优化的优化函数
loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.00001)
# 用cpu计算太慢了，改用GPU计算，model = model.cuda()和X, y = Variable(X.cuda()),
# Variable(y.cuda())就是参与迁移至GPUs的具体代码


print(torch.cuda.is_available())
Use_gpu = torch.cuda.is_available()

if Use_gpu:
    model = model.cuda()
# 开始训练
epoch_n = 5
time_open = time.time()

for epoch in range(epoch_n):
    print('Epoch {}/{}'.format(epoch, epoch_n - 1))
    print('-' * 10)

    for phase in ['train', 'valid']:
        if phase == 'train':
            print('Training...')
            model.train(True)
        else:
            print('Validing...')
            model.train(False)

        running_loss = 0.0
        running_correct = 0.0

        for batch, data in enumerate(dataloader[phase], 1):
            X, Y = data
            if Use_gpu:
                X, Y = Variable(X.cuda()), Variable(Y.cuda())
            else:
                X, Y = Variable(X), Variable(Y)

            y_pred = model(X)

            _, pred = torch.max(y_pred.data , 1)  # 找出每一行中的最大的值对应的索引

            optimizer.zero_grad()

            loss = loss_f(y_pred, Y)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.data.item()
            running_correct += torch.sum(pred == Y.data)

            if batch % 500 == 0 and phase == 'train':
                print('Batch {}, Train Loss:{:.4f},Train ACC: {:.4f}'.format(batch,
                                                                             running_loss / batch,
                                                                             100 * running_correct / (16 * batch)))
        epoch_loss = running_loss * 16 / len(image_datasets[phase])
        epoch_acc = 100 * running_correct / len(image_datasets[phase])
        print('{} Loss:{:.4f} ACC:{:.4f}%'.format(phase, epoch_loss, epoch_acc))
    time_end = time.time() - time_open
    print(time_end)
