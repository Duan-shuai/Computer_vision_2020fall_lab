import numpy as np
# cv2仅用于io读写
import cv2
import my_dft
import matplotlib.pyplot as plt


class HomomorphicFilter:
    # 对于一幅光照不均匀的图像，同态滤波可同时实现亮度调整和对比度提升，从而改善图像质量。
    # 为了压制低频的亮度分量，增强高频的反射分量，滤波器H应是一个高通滤波器，但又不能完全cut off 低频分量，仅作适当压制。

    # Filters:常用butterworth 和 gaussian
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        # D0：params[0] = 30 ，n = 1
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = 1 / (1 + (Duv / filter_params[0] ** 2) ** filter_params[1])
        return (1 - H)
    def __gaussian_filter(self, I_shape, filter_params):

        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        # D0：params[0] = 30 , n = 1
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = np.exp((-Duv / (2 * (filter_params[0]) ** 2)))
        return (1 - H)

    # 频域内处理：I*(a*H + b)
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        # H = my_dft.shift_ft(H)
        # 不同参数结果不同
        I_filtered = (0.75 + 1.25*H) * I
        return I_filtered

    def filter(self, I, filter_params, filter='gaussian', H=None):

        # log + fft(空域 -> 频域)
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)
        # I_fft = my_dft.dft(I_log)

        # 频域中传递函数 H
        if filter == 'butterworth':
            H = self.__butterworth_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'gaussian':
            H = self.__gaussian_filter(I_shape=I_fft.shape, filter_params=filter_params)

        # 频域处理
        I_fft_filt = self.__apply_filter(I=I_fft, H=H)

        # 逆傅里叶变换 + exp
        # I_filt = my_dft.shift_ift(I_fft_filt)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))
        return np.uint8(I)


# img_path_in = '2.0.PNG'
# img_path_in = '2.0.bmp'
img_path_in = '2.0.jpg'
img_path_out = '2.2filtered.bmp'

img = cv2.imread(img_path_in)
cv2.imwrite('2.1gray.bmp', img[:, :, 0])
homo_filter = HomomorphicFilter()
img_filtered = homo_filter.filter(I=img[:, :, 0], filter_params=[500, 2])
cv2.imwrite(img_path_out, img_filtered)

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(img_filtered)
plt.show()
# img0 = cv2.imread('5.PNG')[:, :, 0]
# cv2.imwrite('2.2filtered.bmp', img0)