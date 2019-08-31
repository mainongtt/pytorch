# 这部分是利用pytorch 进行实战，利用迁移vgg16 来实现图片的风格迁移
import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
import copy
from PIL import Image
import matplotlib.pyplot as plt
# %matplotlib inline
# 数据预处理，加载数据
transform = transforms.Compose([transforms.Resize([224,224]),
                               transforms.ToTensor()])

def loading(path = None):
    img = Image.open(path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    return img

content_img = loading('images/4.jpg')
content_img = Variable(content_img).cuda()
style_img = loading('images/5.jpg')
style_img = Variable(style_img).cuda()


# 内容度量值可以使用均方误差作为损失函数：
class Content_loss(torch.nn.Module):
    def __init__(self, weight, target):
        super(Content_loss, self).__init__()
        self.weight = weight  # weight 是一个权重用来控制最后风格和内容的占比
        self.target = target.detach() * weight  # target是通过卷积获取到的输入图像中的内容,target.detach()用于对提取到的内容进行锁定，不需要进行梯度
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, input):
        self.loss = self.loss_fn(input * self.weight, self.target)  # 进行计算损失值
        return input

    def backward(self):
        self.loss.backward(retain_graph=True)  # 将损失值进行后向传播
        return self.loss


# 风格度量使用均方误差作为损失函数：
class Gram_matrix(torch.nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())  # 这里涉及格拉姆矩阵，主要的作用就是在计算风格损失之前，先对风格图像的风格通过矩阵内积使得风格变得更加突出
        return gram.div(a * b * c * d)


class Style_loss(torch.nn.Module):
    def __init__(self, weight, target):
        super(Style_loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss_fn = torch.nn.MSELoss()
        self.gram = Gram_matrix()

    def forward(self, input):
        self.Gram = self.gram(input.clone())
        self.Gram.mul_(self.weight)
        self.loss = self.loss_fn(self.Gram, self.target)
        return input

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss
# 用cpu计算太慢了，改用GPU计算，model = model.cuda()和X, y = Variable(X.cuda()),
# Variable(y.cuda())就是参与迁移至GPUs的具体代码


print(torch.cuda.is_available())
use_gpu = torch.cuda.is_available()
cnn = models.vgg16(pretrained=True).features  # 这里只迁移VGG16的特征提取的部分层

if use_gpu:
    cnn = cnn.cuda()

model = copy.deepcopy(cnn)

content_layer = ['Conv_3']  # 定义这一层用来提取内容

style_layer = ['Conv_1', 'Conv_2', 'Conv_3', 'Conv_4']  # 定义这几层用来提取风格

content_losses = []
style_losses = []

content_weight = 1  # 分别定义内容、风格的比重
style_weight = 1000

# 搭建图像风格迁移模型
new_model = torch.nn.Sequential()
# model = copy.deepcopy(cnn)
gram = Gram_matrix()

if use_gpu:
    new_model = new_model.cuda()
    gram = gram.cuda()

index = 1

# 这里由于我们的定义，对于迁移的模型，只需要用到前8层即可完成特征的提取，利用add_module来构建一个新的自定义模型
for layer in list(model)[:8]:
    if isinstance(layer, torch.nn.Conv2d):
        name = 'Conv_' + str(index)
        new_model.add_module(name, layer)
        if name in content_layer:
            target = new_model(content_img).clone()
            content_loss = Content_loss(content_weight, target)
            new_model.add_module('content_loss' + str(index), content_loss)
            content_losses.append(content_loss)

        if name in style_layer:
            target = new_model(style_img).clone()
            target = gram(target)
            style_loss = Style_loss(style_weight, target)
            new_model.add_module('style_loss_' + str(index), style_loss)
            style_losses.append(style_loss)

    if isinstance(layer, torch.nn.ReLU):
        name = 'ReLU_' + str(index)
        layer = torch.nn.ReLU(
            inplace=False)  # 对于pytorch0.4以后的版本，不支持inplace=True的操作，因此需要重写为inplace = False，如果是0.3版就可以不用这一句
        new_model.add_module(name, layer)
        index = index + 1

    if isinstance(layer, torch.nn.MaxPool2d):
        name = 'MaxPool_' + str(index)
        new_model.add_module(name, layer)

print(new_model)
# 参数优化
input_img = content_img.clone()
parameter = torch.nn.Parameter(input_img.data)
optimizer = torch.optim.LBFGS([parameter]) # 在这个模型中需要优化的损失值有多个并且规模较大，使用该优化函数可以取得更好的效果。
# 接下来可以进行模型的训练了
epoch_n = 300
epoch = [0]
while epoch[0] <= epoch_n:

    def closure():
        optimizer.zero_grad()
        style_score = 0
        content_score = 0
        parameter.data.clamp_(0, 1)
        new_model(parameter)
        for sl in style_losses:
            style_score += sl.backward()

        for cl in content_losses:
            content_score += cl.backward()

        epoch[0] += 1
        if epoch[0] % 50 == 0:
            print('Epoch:{} Style_loss: {:4f} Content_loss: {:.4f}'.format(epoch[0], style_score.data.item(),
                                                                           content_score.data.item()))
        return style_score + content_score

    optimizer.step(closure)
output = parameter.data
unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()
plt.figure()
def imshow(tensor, title=None):
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(3, 224, 224)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()
