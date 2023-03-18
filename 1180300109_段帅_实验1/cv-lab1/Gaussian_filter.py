from ReadBmp import ReadBmp
import math
import numpy as np

class Gaussian_filter:
    def getamat(self, filename):
        bmp = ReadBmp(filename)
        temp = []
        for i in range(bmp.biHeight*bmp.biWidth):
            # 图像灰度化
            gray = round(0.299 * bmp.data[i][0] + 0.578 * bmp.data[i][1] + 0.114 * bmp.data[i][2])
            temp.append(gray)
        temp1 = np.array(temp)
        matdata = temp1.reshape(bmp.biHeight, bmp.biWidth)
        return matdata
    def getabmp(self, gray, filename1, filename2):
        bmp = ReadBmp(filename1)
        c = 0
        # gray = np.array()
        m, n = gray.shape
        new_gray = gray.tolist()
        for i in range(m):
            for j in range(n):
                bmp.data[c][0] = int(new_gray[i][j])
                bmp.data[c][1] = int(new_gray[i][j])
                bmp.data[c][2] = int(new_gray[i][j])
                c = c + 1
        print(c)
        bmp.creataBmp(filename2)
    def gaussiankernel(self, W, H, gray):
        # 归一化 与 方差
        sigma1 = sigma2 = 1
        sum = 0

        gaussian = np.zeros([5, 5])
        for i in range(5):
            for j in range(5):
                gaussian[i, j] = math.exp(-1 / 2 * (np.square(i - 3) / np.square(sigma1)  # 生成二维高斯分布矩阵
                                                    + (np.square(j - 3) / np.square(sigma2)))) / (
                                         2 * math.pi * sigma1 * sigma2)
                sum = sum + gaussian[i, j]
        gaussian = gaussian / sum
        # print(gaussian)

        # step1.高斯滤波
        new_gray = np.zeros([W, H])
        for i in range(W - 5):
            for j in range(H - 5):
                new_gray[i, j] = np.sum(gray[i:i + 5, j:j + 5] * gaussian)  # 与高斯矩阵卷积实现滤波
        return new_gray