#  这部分是关于pytorch的基本知识
import torch
a = torch.FloatTensor(2,3)    # pytorch 定义数据类型的方式，可以输入一个维度值或者列表
b = torch.FloatTensor([2,3,4,5])
# print(a)
# print(b)
c = torch.IntTensor(2,3)
d = torch.IntTensor([2,3,4,5])
print(c)
print(d)
e = torch.rand(2,3)  # 随机生成的浮点数据在0～1区间均匀分布
f = torch.randn(2,3) # 随机生成的浮点数的取值满足均值为0、方差为1的正太分布
g = torch.arange(1,20,1) # 用于生成数据类型为浮点型且自定义起始范围和结束范围(注意是前闭后开集)，参数有三个，分别是范围的起始值、结束值和步长
h = torch.zeros(2,3)  # 浮点型的Tensor中的元素值全部为0
print(e)
print(f)
print(g)
print(h)

a = torch.randn(2,3)
print(a)
b = torch.abs(a)
print(b)
c = torch.randn(2,3)
d = torch.add(a,b)
print(d)
e = torch.add(d,10)
print(e)
f = torch.clamp(d,-0.1,0.1) # 对tensor类型的变量进行裁剪，裁剪范围为(-0.1,0.1),即将元素的值裁剪到指定的范围内
print(f)
g = torch.div(a,b)   #   除法
print(g)
h = torch.div(g,2)
print(h)
j = torch.mul(g,h)    # 乘法
print(j)
k = torch.mul(j,10)
print(k)
l = torch.pow(j,2)   # 求幂次方
print(l)
q = torch.pow(k,l)
print(q)
w = torch.randn(2,3)
print(w)
e = torch.randn(3,2)
print(e)
r = torch.mm(w,e)   #  矩阵乘法，输入的维度需要满足矩阵乘法
print(r)
t = torch.randn(3)
print(t)
y = torch.mv(w,t)  # 矩阵与向量之间的乘法规则进行计算，被传入的参数中的第1个参数代表矩阵，第2个参数代表向量，顺序不能颠倒
print(y)
