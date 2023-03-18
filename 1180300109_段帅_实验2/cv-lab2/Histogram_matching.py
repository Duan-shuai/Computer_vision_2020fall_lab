from ReadBmp import ReadBmp
import matplotlib.pyplot as plt
import numpy as np
import math


def get_matching_array():
    tmp = []
    for i in range(256):
        if i <= 50:
            tmp.append(1)
        elif i <= 200:
            tmp.append(2)
        else:
            tmp.append(1)
    h = np.array(tmp)
    h = h / 305
    return h


# 1.原图像直方图均衡化--f
# 2.规定的直方图均衡化--g
# 3.构建f->g

# 1.1 获取归一化的规定直方图 h1
filename1 = "1.bmp"
bmp = ReadBmp(filename1)
bmp.gray()
bmp.creataBmp("1.1gray.bmp")

h1 = get_matching_array()

# 1.2 规定直方图的均衡化
hp1 = np.array([0.0 for i in range(256)])
print(h1)
for i in range(256):
    hp1[i] = np.round(np.sum(h1[0:i + 1]) * 255)
# 不进行溢出处理，使用uint8转化时有问题
    if hp1[i] < 0:
        hp1[i] = 0
    elif hp1[i] > 255:
        hp1[i] = 255
print(hp1)
# (0~255整数)
T1 = hp1.astype('uint8')
print(T1)

# 2.原图像均衡化
# 2.1 统计各灰度级像素数--h[256]
h = np.array([0 for i in range(256)])
for pixel in bmp.data:
    # 三个通道p[0],p[1],p[2]均灰度化
    h[pixel[0]] = h[pixel[0]] + 1
# 2.2 归一化--hs （ P(f) = nj/n )
hs = h / len(bmp.data)

# 2.3 计算累计分布--hp ( C(f) )
# 2.4 计算映射后的g = （255 - 0）*C（f）（取整数）-- T， 得到新灰度级
hp = np.array([0.0 for i in range(256)])
for i in range(256):
    hp[i] = np.round(np.sum(hs[0:i + 1]) * 255)
# (0~255整数)
T2 = hp.astype('uint8')

# print(h1, hs)
# 3. 构建映射T: T2 -> T1
# 使用SML
T = np.zeros(256)
for i in range(256):
    tmp = T2[i]
    v = T1[tmp]
    T[i] = v
T = T.astype('uint8')
print(T, T1, T2)

for pixel in bmp.data:
    # T为原灰度到现灰度的映射
    s = T[pixel[0]]
    pixel[0] = s
    pixel[1] = s
    pixel[2] = s
bmp.creataBmp("1.3.bmp")

# print(h)

# plt.subplot(1, 2, 1)
# plt.plot(h)
# plt.show()
