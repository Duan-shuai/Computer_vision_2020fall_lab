from ReadBmp import ReadBmp
import matplotlib.pyplot as plt
import numpy as np

filename1 = "1.bmp"
bmp = ReadBmp(filename1)
bmp.gray()

# 1. 统计各灰度级像素数--h[256]
h = np.array([0 for i in range(256)])
# 像素list -- h1[n*m]
h1 = []
for pixel in bmp.data:
    # 三个通道p[0],p[1],p[2]均灰度化
    h[pixel[0]] = h[pixel[0]] + 1
    h1.append(pixel[0])
# 2. 归一化--hs （ P(f) = nj/n )
hs = h / len(bmp.data)

# 3. 计算累计分布--hp ( C(f) )
# 4. 计算映射后的g = （255 - 0）*C（f）（取整数）-- T， 得到新灰度级
hp = np.array([0.0 for i in range(256)])
for i in range(256):
    hp[i] = np.round(np.sum(hs[0:i+1]) * 255)
# (0~255整数)
T = hp.astype('uint8')

# 5. 再统计新灰度级下像素数--h2
# 根据映射T得到新的图像
hn = np.array([0 for i in range(256)])
h2 = []
for pixel in bmp.data:
    # T为原灰度到现灰度的映射
    s = T[pixel[0]]
    pixel[0] = s
    pixel[1] = s
    pixel[2] = s
    hn[pixel[0]] = hn[pixel[0]] + 1
    h2.append(s)
bmp.creataBmp("1.2.bmp")

# 画出原先的直方图
# 1行2列的画布 ，处理第1个画布
plt.subplot(1, 2, 1)
plt.hist(h1, bins = 256)

# 画出新图像的直方图
plt.subplot(1,2,2)
plt.hist(h2, bins = 256)
plt.show()