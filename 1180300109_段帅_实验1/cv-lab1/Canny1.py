from ReadBmp import ReadBmp
import numpy as np
import matplotlib.pyplot as plt
import math
from Gaussian_filter import Gaussian_filter

class Canny1:
    def canny(self, filename1, filename2):

        bmp = ReadBmp(filename1)
        for i in range(bmp.biHeight * bmp.biWidth):
            # 图像灰度化
            gray1 = round(0.299 * bmp.data[i][0] + 0.578 * bmp.data[i][1] + 0.114 * bmp.data[i][2])
            bmp.data[i][0] = gray1
            bmp.data[i][1] = gray1
            bmp.data[i][2] = gray1

        # step1.高斯平滑
        g = Gaussian_filter()
        gray = g.getamat(filename1)
        W, H = gray.shape
        # print(W, H) #- 40900 3
        new_gray = np.zeros([W, H])
        new_gray = g.gaussiankernel(W, H, gray)

        # step2.增强 通过求梯度幅值（Roberts算子）
        W1, H1 = new_gray.shape
        dx = np.zeros([W1, H1])
        dy = np.zeros([W1, H1])
        d = np.zeros([W1, H1])
        for i in range(W1 - 1):
            for j in range(H1 - 1):
                # x方向梯度
                dx[i, j] = new_gray[i, j + 1] - new_gray[i, j]
                # y方向梯度
                dy[i, j] = new_gray[i + 1, j] - new_gray[i, j]
                # 梯度幅值
                d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))  # 图像梯度幅值作为图像强度值
                if d[i][j] > 255:
                    d[i][j] = 255
                if d[i][j] < 0:
                    d[i][j] = 0
        g.getabmp(d, "1.bmp", "step2.bmp")

        # setp3.非极大值抑制 NMS
        W2, H2 = d.shape
        NMS = np.copy(d)
        NMS[0, :] = NMS[W2 - 1, :] = NMS[:, 0] = NMS[:, H2 - 1] = 0
        for i in range(1, W2 - 1):
            for j in range(1, H2 - 1):

                if d[i, j] == 0:
                    NMS[i, j] = 0
                else:
                    gradX = dx[i, j]
                    gradY = dy[i, j]
                    gradTemp = d[i, j]

                    if np.abs(gradY) > np.abs(gradX):
                        weight = np.abs(gradX) / np.abs(gradY)
                        grad2 = d[i - 1, j]
                        grad4 = d[i + 1, j]
                        if gradX * gradY > 0:
                            grad1 = d[i - 1, j - 1]
                            grad3 = d[i + 1, j + 1]
                        else:
                            grad1 = d[i - 1, j + 1]
                            grad3 = d[i + 1, j - 1]

                    else:
                        weight = np.abs(gradY) / np.abs(gradX)
                        grad2 = d[i, j - 1]
                        grad4 = d[i, j + 1]
                        if gradX * gradY > 0:
                            grad1 = d[i + 1, j - 1]
                            grad3 = d[i - 1, j + 1]
                        else:
                            grad1 = d[i - 1, j - 1]
                            grad3 = d[i + 1, j + 1]

                    gradTemp1 = weight * grad1 + (1 - weight) * grad2
                    gradTemp2 = weight * grad3 + (1 - weight) * grad4
                    if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                        NMS[i, j] = gradTemp
                    else:
                        NMS[i, j] = 0

        g.getabmp(NMS, "1.bmp", "step3.bmp")

        # step4. 双阈值算法检测、连接边缘
        W3, H3 = NMS.shape
        DT = np.zeros([W3, H3])
        # 定义高低阈值
        TL = 0.2 * np.max(NMS)
        TH = 0.3 * np.max(NMS)
        for i in range(1, W3 - 1):
            for j in range(1, H3 - 1):
                if (NMS[i, j] < TL):
                    DT[i, j] = 0
                elif (NMS[i, j] > TH):
                    DT[i, j] = 255
                elif ((NMS[i - 1, j - 1:j + 1] < TH).any() or (NMS[i + 1, j - 1:j + 1]).any()
                      or (NMS[i, [j - 1, j + 1]] < TH).any()):
                    DT[i, j] = 255
        g.getabmp(DT, "1.bmp", "step4.bmp")
        # plt.imshow(DT, cmap="gray")