from struct import unpack
from struct import pack

class ReadBmp:
    def __init__(self, filename):
        with open(filename, 'rb') as f:
            # DWORD是32位无符号整数
            # 读取bmp文件头，14字节
            self.bfType = unpack("<h", f.read(2))[0] # 说明文件类型
            self.bfSize = unpack("<i", f.read(4))[0] # 说明位图大小，以字节为单位
            self.bfReserved1 = unpack("<h", f.read(2))[0] # 保留位，设为零
            self.bfReserved2 = unpack("<h", f.read(2))[0] # 保留位，设为零
            self.bfOffBits = unpack("<i", f.read(4))[0] # 从文件头实际图像数据之间的字节偏移量
            # 读取bmp文件位图信息头，40字节
            self.biSize = unpack("<i", f.read(4))[0]  # 说明位图信息头大小
            self.biWidth = unpack("<i", f.read(4))[0]  # 说明图像的宽度，单位为像素
            self.biHeight = unpack("<i", f.read(4))[0]  # 说明图像的高度，单位为像素，为正时表示数据排列是从左下按行到右上
            self.biPlanes = unpack("<h", f.read(2))[0]  # 说明颜色平面数 总设为 1
            self.biBitCount = unpack("<h", f.read(2))[0]  # 说明比特数/像素
            self.biCompression = unpack("<i", f.read(4))[0]  # 图像压缩的数据类型
            self.biSizeImage = unpack("<i", f.read(4))[0]  # 图像大小，单位为字节
            self.biXPelsPerMeter = unpack("<i", f.read(4))[0]  # 水平分辨率，像素/米
            self.biYPelsPerMeter = unpack("<i", f.read(4))[0]  # 垂直分辨率，像素/米
            self.biClrUsed = unpack("<i", f.read(4))[0]  # 说明实际使用的彩色表中的颜色索引数
            self.biClrImportant = unpack("<i", f.read(4))[0]  # 说明对图像显示有重要影响的颜色索引的数目

            # 经查询，图像大小128*128，24位，文件大小49206字节 = 14 + 40 + 128*128*3

            # 只能读取24位的图像
            # 格式设置：data里按像素点储存，一个像素点为一个数组
            self.data = []
            for i in range(self.biHeight):
                count = 0
                for j in range(self.biWidth):
                    pixel = []
                    pixel.append(unpack("<B", f.read(1))[0])
                    pixel.append(unpack("<B", f.read(1))[0])
                    pixel.append(unpack("<B", f.read(1))[0])
                    self.data.append(pixel)
                    count = count + 3
                # 四字节对齐，把末尾补的零删去
                while count % 4 != 0:
                    f.read(1)
                    count = count + 1

        f.close()

    def creataBmp(self, filename):
        # 用已有数据创建一个bmp图像并输出，并输出到指定路径
        file = open(filename, 'wb+')
        file.write(pack("<h", self.bfType))
        file.write(pack("<i", self.bfSize))
        file.write(pack("<h", self.bfReserved1))
        file.write(pack("<h", self.bfReserved2))
        file.write(pack("<i", self.bfOffBits))

        file.write(pack("<i", self.biSize))
        file.write(pack("<i", self.biWidth))
        file.write(pack("<i", self.biHeight))
        file.write(pack("<h", self.biPlanes))
        file.write(pack("<h", self.biBitCount))
        file.write(pack("<i", self.biCompression))
        file.write(pack("<i", self.biSizeImage))
        file.write(pack("<i", self.biXPelsPerMeter))
        file.write(pack("<i", self.biYPelsPerMeter))
        file.write(pack("<i", self.biClrUsed))
        file.write(pack("<i", self.biClrImportant))

        count = 0
        for pixel in self.data:
            file.write(pack("<B", pixel[0]))
            file.write(pack("<B", pixel[1]))
            file.write(pack("<B", pixel[2]))
            count = count + 3
            # 四字节对齐
            if count == self.biWidth*3:
                while(count % 4 != 0):
                    file.write(pack("<B", 0))
                count = 0
        file.close()
