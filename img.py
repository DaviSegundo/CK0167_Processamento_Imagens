from PIL import Image, ImageFilter
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import functions as fc


class Img():

    def __init__(self, path):
        self.path = path
        self.img = Image.open(path)
        self.img_uint8 = np.array(self.img)
        self.img_now = self.img_uint8
        self.return_img = None

    def size(self):
        return self.img.size

    def show_img_now(self):
        return Image.fromarray(self.img_now)

    def histogram_plot(self):
        img = Image.fromarray(self.img_now)
        hist = img.histogram()
        for i in range(0, 255):
            plt.bar(i, hist[i], color='gray')
        plt.title('Histogram', size=15)

        strfile = './media/hist.png'
        plt.savefig(strfile, dpi=100)
        plt.close()
        return Image.open(strfile)

    # funções para testar os resultados dos parâmetros
    def negative_image_test(self, array):
        temp_img = array/255
        temp_img = 1 - temp_img
        temp_img = np.clip(temp_img, 0, 1)
        temp_img = (temp_img * 255).astype(np.uint8)
        self.return_img = temp_img
        return self.return_img

    def brightness_test(self, array, num):
        temp_img = array/255
        temp_img = temp_img*num
        temp_img = np.clip(temp_img, 0, 1)
        temp_img = (temp_img * 255).astype(np.uint8)
        self.return_img = temp_img
        return self.return_img

    def gama_test(self, array, num):
        temp_img = array/255
        temp_img = temp_img**num
        temp_img = np.clip(temp_img, 0, 1)
        temp_img = (temp_img * 255).astype(np.uint8)
        self.return_img = temp_img
        return self.return_img

    def log_test(self, array):
        temp_img = array/255
        temp_img = np.log2(1 + temp_img)
        temp_img = np.clip(temp_img, 0, 1)
        temp_img = (temp_img * 255).astype(np.uint8)
        self.return_img = temp_img
        return self.return_img

    def equalize_hist_test(self, array):
        temp_img = array
        temp_img = Image.fromarray(temp_img)
        hist_temp_img = temp_img.histogram()
        size_temp_img = temp_img.size
        pixels = size_temp_img[0] * size_temp_img[1]
        temp_img = array

        hist_norm = np.array(hist_temp_img)/pixels
        hist_sum = hist_norm.copy()

        sum = 0
        for i, j in enumerate(hist_norm):
            sum += j
            hist_sum[i] = sum

        equal_hist = hist_sum * 255

        for i, j in enumerate(equal_hist):
            equal_hist[i] = int(j)

        for i in range(len(temp_img)):
            for j in range(len(temp_img[i])):
                temp_img[i][j] = equal_hist[temp_img[i][j]]

        temp_img = (temp_img).astype(np.uint8)
        self.return_img = temp_img
        return self.return_img

    def linear_parts_test(self, points_x, points_y, array):
        temp_img = array/255
        for i in range(len(temp_img)):
            for j in range(len(temp_img[i])):
                temp_img[i][j] = fc.linear_parts(points_x, points_y, temp_img[i][j])
        temp_img = (temp_img * 255).astype(np.uint8)
        self.return_img = temp_img
        print('linear test')
        return self.return_img

    def laplacian_filter_test(self, array):
        laplacian = np.array([[0,1,0], 
                              [1,-4,1], 
                              [0,1,0]])
        temp_img = array
        temp_img = convolve2d(abs(temp_img), laplacian)
        temp_img = fc.normalize_img(temp_img)
        self.return_img = temp_img
        return self.return_img

    def high_boost_filter_test(self, array):
        gaussian = np.array([[1,2,1],
                             [2,8,2],
                             [1,2,1]])
        temp_img = array
        ipg = convolve2d(abs(temp_img), gaussian)
        shp = ipg.shape
        ipg = ipg[0:shp[0]-2, 0:shp[1]-2]
        ib = temp_img - ipg
        self.return_img = (ib*15)
        return self.return_img

    def mean_simple_filter_test(self, array, size):
        kernel = fc.generate_mean_simple_kernel(size)
        temp_img = array
        temp_img = convolve2d(abs(temp_img), kernel)
        temp_img = fc.normalize_img(temp_img)
        self.return_img = temp_img
        return self.return_img

    def mean_weighted_filter_test(self, array, size):
        kernel = fc.generate_mean_weighted_kernel(size)
        temp_img = array
        temp_img = convolve2d(abs(temp_img), kernel)
        temp_img = fc.normalize_img(temp_img)
        self.return_img = temp_img
        return self.return_img

    def median_filter_test(self, array, size):
        temp_img = array
        temp_img = Image.fromarray(temp_img)
        temp_img = temp_img.filter(ImageFilter.MedianFilter(size))
        self.return_img = np.array(temp_img)
        return self.return_img

    def convert(self, array):
        return Image.fromarray(array)

    # aplicação definitiva das informações alteradas
    def negative_image(self):
        temp_img = self.img_now/255
        temp_img = 1 - temp_img
        temp_img = np.clip(temp_img, 0, 1)
        temp_img = (temp_img * 255).astype(np.uint8)
        self.img_now = temp_img
        return Image.fromarray(self.img_now)

    def brightness_apply(self, num):
        temp_img = self.img_now/255
        temp_img = temp_img*num
        temp_img = np.clip(temp_img, 0, 1)
        temp_img = (temp_img * 255).astype(np.uint8)
        self.img_now = temp_img
        return Image.fromarray(self.img_now)

    def gama_apply(self, num):
        temp_img = self.img_now/255
        temp_img = temp_img**num
        temp_img = np.clip(temp_img, 0, 1)
        temp_img = (temp_img * 255).astype(np.uint8)
        self.img_now = temp_img
        return Image.fromarray(self.img_now)

    def log_apply(self):
        temp_img = self.img_now/255
        temp_img = np.log2(1+temp_img)
        temp_img = np.clip(temp_img, 0, 1)
        temp_img = (temp_img * 255).astype(np.uint8)
        self.img_now = temp_img
        return Image.fromarray(self.img_now)

    def equalize_hist(self):
        temp_img = self.img_now
        temp_img = Image.fromarray(temp_img)
        hist_temp_img = temp_img.histogram()
        size_temp_img = temp_img.size
        pixels = size_temp_img[0] * size_temp_img[1]
        temp_img = self.img_now

        hist_norm = np.array(hist_temp_img)/pixels
        hist_sum = hist_norm.copy()

        sum = 0
        for i, j in enumerate(hist_norm):
            sum += j
            hist_sum[i] = sum

        equal_hist = hist_sum * 255

        for i, j in enumerate(equal_hist):
            equal_hist[i] = int(j)

        for i in range(len(temp_img)):
            for j in range(len(temp_img[i])):
                temp_img[i][j] = equal_hist[temp_img[i][j]]

        temp_img = (temp_img).astype(np.uint8)
        self.img_now = temp_img
        return Image.fromarray(self.img_now)

    def linear_parts_apply(self, points_x, points_y):
        temp_img = self.img_now/255
        for i in range(len(temp_img)):
            for j in range(len(temp_img[i])):
                temp_img[i][j] = fc.linear_parts(points_x, points_y, temp_img[i][j])
        temp_img = (temp_img * 255).astype(np.uint8)
        self.img_now = temp_img
        return Image.fromarray(self.img_now)


    def laplacian_filter_apply(self):
        laplacian = np.array([[0,1,0], 
                              [1,-4,1], 
                              [0,1,0]])
        temp_img = self.img_now
        temp_img = convolve2d(abs(temp_img), laplacian)
        shp = self.img_now.shape
        k = temp_img[0:shp[0], 0:shp[1]]
        self.img_now = self.img_now + (0.3*k)
        return Image.fromarray(self.img_now)

    def high_boost_filter_apply(self):
        gaussian = np.array([[1,2,1],
                             [2,8,2],
                             [1,2,1]])
        temp_img = self.img_now
        ipg = convolve2d(abs(temp_img), gaussian)
        ib = temp_img - ipg
        ibn = fc.normalize_img(ib)
        self.return_img = ibn*1.5
        return self.return_img

    def mean_simple_filter_apply(self, size):
        kernel = fc.generate_mean_simple_kernel(size)
        temp_img = self.img_now
        temp_img = convolve2d(abs(temp_img), kernel)
        self.img_now = temp_img
        return Image.fromarray(self.img_now)

    def mean_weighted_filter_apply(self, size):
        kernel = fc.generate_mean_weighted_kernel(size)
        temp_img = self.img_now
        temp_img = convolve2d(abs(temp_img), kernel)
        self.img_now = temp_img
        return Image.fromarray(self.img_now)

    def median_filter_apply(self, size):
        temp_img = self.img_now
        temp_img = Image.fromarray(temp_img)
        temp_img = temp_img.filter(ImageFilter.MedianFilter(size))
        self.img_now = np.array(temp_img)
        return Image.fromarray(self.img_now)


if __name__ == "__main__":
    laplacian = np.array([[1,1,1], [1,-8.5,1], [1,1,1]])
    # h_sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # v_sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    from sklearn.preprocessing import normalize

    img = Image.open('../../Downloads/Imagens_PDI/DIP3E_Original_Images_CH03/Fig0354(a)(einstein_orig).tif')
    img.show()

    array = np.array(img)
    print(array)

    temp_img = convolve2d(array, laplacian)
    print(temp_img)

    norm_img = np.interp(temp_img, (temp_img.min(), temp_img.max()), (0, 1))
    print(norm_img)

    norm_img = norm_img*255
    norm_img = Image.fromarray(norm_img)
    norm_img.show()

    


