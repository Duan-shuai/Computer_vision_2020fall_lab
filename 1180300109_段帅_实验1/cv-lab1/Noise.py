from ReadBmp import ReadBmp
import random

class Noise:
    def GaussianNoise(self, filename1, filename2):
        bmp = ReadBmp(filename1)
        # 生成均值为means，方差为sigma，噪声比小于percetage的高斯噪声图像
        means = 0
        sigma = 10
        percetage = 0.5
        for pixel in bmp.data:
            # 图像灰度化
            gray = 0.299 * pixel[0] + 0.578 * pixel[1] + 0.114 * pixel[2]
            # 随机产生噪声
            p = random.random()
            # 增加高斯噪声
            if p <= percetage:
                gray = gray + random.gauss(means, sigma)
            # 灰度越界处理
            if gray < 0:
                gray = 0
            elif gray > 255:
                gray = 255

            pixel[0] = round(gray)
            pixel[1] = round(gray)
            pixel[2] = round(gray)
        # 将新图像保存到文件
        bmp.creataBmp(filename2)
        return

    def SaltAndPepperNoise(self, filename1, filename2):
        bmp = ReadBmp(filename1)
        # 生成椒盐噪声图像
        # 信噪比：含有噪声的像素块比例
        percetage = 0.1
        for pixel in bmp.data:
            # 图像灰度化
            gray = 0.299 * pixel[0] + 0.578 * pixel[1] + 0.114 * pixel[2]
            p = random.random()
            # 增加椒盐噪声
            if p <= percetage:
                # （0，0.025）盐粒噪声
                if p <= percetage * 0.5:
                    gray = 0
                # （0.025，0.05）胡椒噪声
                else:
                    gray = 255

            pixel[0] = round(gray)
            pixel[1] = round(gray)
            pixel[2] = round(gray)

        # 将新图像保存到文件
        bmp.creataBmp(filename2)
        return