from Canny1 import Canny1
from Gaussian_filter import Gaussian_filter
from ReadBmp import ReadBmp
from Noise import Noise
from WaveFilter import WaveFilter
from EdgeDetection import EdgeDetection
import numpy as np


noise = Noise()
# 创建高斯噪声
noise.GaussianNoise("1.gray.bmp", "GaussianNoise.bmp")
# 创建椒盐噪声
noise.SaltAndPepperNoise("1.bmp", "SaltAndPepperNoise.bmp")

# 对高斯噪声图像进行均值滤波
filter = WaveFilter()
filter.MeanFilter("GaussianNoise.bmp", "MeanFilter.bmp")

# 对椒盐噪声图像进行中值滤波
filter.MedianFilter("SaltAndPepperNoise.bmp", "MedianFilter.bmp")

# Sobel算子边缘检测
detection = EdgeDetection()
detection.Sobel("1.bmp", "sobel.bmp")

# Roberts算子边缘检测
# d = Canny1()
detection.Roberts("1.bmp", "roberts.bmp")

# Laplacian算子边缘检测
detection.Laplacian("1.bmp", "laplacian.bmp")


# Canny算子
d = Canny1()
d.canny("sobel.bmp", "Canny.bmp")

# 高斯滤波-平滑处理
g = Gaussian_filter()
gray = g.getamat("1.bmp")
W, H = gray.shape
print(W, H)  #- 40900 3
new_gray = np.zeros([W, H])
new_gray = g.gaussiankernel(W, H, gray)
g.getabmp(new_gray, "1.bmp", "Gaussian_filter.bmp")
