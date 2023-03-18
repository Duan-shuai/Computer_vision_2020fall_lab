import numpy as np
import cv2
import math


FILTER_DIAMETER = 1
SIGMA_R = 30
SIGMA_D = 3
PROC_NUM = 4

# 距离
def distance(x, y, i, j):
    return np.sqrt((x - i) ** 2 + (y - j) ** 2)

# 高斯
def gaussian(x, sigma):
    return (1 / (2 * math.pi * (sigma ** 2))) * math.exp(-(x ** 2) / (2 * (sigma ** 2)))


def apply_bilateral_filter(source, filtered_image, row, col, diameter, sigma_r, sigma_d):
    hl = int(diameter / 2)
    i_filtered = 0
    wp = 0
    i = 0
    while i < diameter:
        j = 0
        neighbour_row = int(row - (hl - i))
        if 0 <= neighbour_row < len(source):
            while j < diameter:
                neighbour_col = int(col - (hl - j))
                if 0 <= neighbour_col < len(source):
                    # （空间核）
                    gauss_r = gaussian(abs(source[row][col] - source[neighbour_row][neighbour_col]), sigma_r)
                    # （像素核）
                    gauss_d = gaussian(distance(row, col, neighbour_row, neighbour_col), sigma_d)
                    # 双边滤波的核
                    w = gauss_r * gauss_d
                    # 卷积
                    i_filtered += w * source[neighbour_row][neighbour_col]
                    wp += w
                j += 1
        i += 1
    i_filtered = i_filtered / wp
    filtered_image[row][col] = int(i_filtered)


def bilateral_filter_own(args):
    source, filter_diameter, sigma_r, sigma_d = args
    # 目标位置
    filtered_image = np.zeros(source.shape)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_r, sigma_d)
            j += 1
        i += 1
    return filtered_image


if __name__ == "__main__":
    img = cv2.imread('1.bmp')[:, :, 0]
    grey_img = bilateral_filter_own(args=[img, FILTER_DIAMETER, SIGMA_R, SIGMA_D])
    cv2.imwrite('3.1bilateral.bmp', grey_img)
