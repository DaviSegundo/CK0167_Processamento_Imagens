from PIL import Image, ImageFilter, ImageEnhance
import os
import io
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import functions as fc
import cv2
from math import ceil
from skimage.color import rgb2hsv, hsv2rgb
import colorsys


class Img():

    def __init__(self, path):
        self.path = path
        self.img = Image.open(path)
        self.img_uint8 = np.array(self.img)
        self.img_now = self.img_uint8
        self.img_original = self.img_now
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
                temp_img[i][j] = fc.linear_parts(
                    points_x, points_y, temp_img[i][j])
        temp_img = (temp_img * 255).astype(np.uint8)
        self.return_img = temp_img
        return self.return_img

    def laplacian_filter_test(self, array):
        laplacian = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])
        if len(array.shape) < 3:
            temp_img = array
            shp = array.shape
            ipg = convolve2d(abs(temp_img), laplacian)
            ipg = ipg[0:shp[0], 0:shp[1]]
            ib =  ipg
            ib = ib[0:shp[0], 0:shp[1]]
            self.return_img = fc.normalize_img(1.5*ib)
            return self.return_img
        else:
            img_hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
            img_hsv[:,:,2] = (convolve2d(img_hsv[:,:,2], laplacian, mode="same"))*0.3
            img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            self.return_img = fc.normalize_img(((img_rgb)*0.3)/255)
            return self.return_img

    def high_boost_filter_test(self, array):
        gaussian = np.array([[1, 2, 1],
                             [2, -8, 2],
                             [1, 2, 1]])
        if len(array.shape) < 3:
            temp_img = array
            shp = array.shape
            ipg = convolve2d(abs(temp_img), gaussian)
            ipg = ipg[0:shp[0], 0:shp[1]]
            ib = temp_img - ipg
            ib = ib[0:shp[0], 0:shp[1]]
            self.return_img = fc.normalize_img(3*ib)
            return self.return_img
        else:
            img_hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
            img_hsv[:,:,2] = (convolve2d(img_hsv[:,:,2], gaussian, mode="same"))*0.3
            img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            self.return_img = fc.normalize_img(((img_rgb)*0.3)/255)
            return self.return_img

    def sobel_x_filter_test(self, array):
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        temp_img = array
        temp_img = convolve2d(abs(temp_img), sobel_x)
        temp_img = fc.normalize_img(temp_img)
        self.return_img = temp_img
        return self.return_img

    def sobel_y_filter_test(self, array):
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        temp_img = array
        temp_img = convolve2d(abs(temp_img), sobel_y)
        temp_img = fc.normalize_img(temp_img)
        self.return_img = temp_img
        return self.return_img

    def mean_simple_filter_test(self, array, size):
        kernel = fc.generate_mean_simple_kernel(size)
        if len(array.shape) < 3:
            temp_img = array
            temp_img = convolve2d(abs(temp_img), kernel, mode="same")
            temp_img = fc.normalize_img(temp_img)
            self.return_img = temp_img
            return self.return_img
        else:
            img_hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
            img_hsv[:,:,2] = convolve2d(img_hsv[:,:,2], kernel, mode="same")
            img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            self.return_img = img_rgb
            return self.return_img

    def mean_weighted_filter_test(self, array, size):
        kernel = fc.generate_mean_weighted_kernel(size)
        if len(array.shape) < 3:
            temp_img = array
            temp_img = convolve2d(abs(temp_img), kernel, mode="same")
            temp_img = fc.normalize_img(temp_img)
            self.return_img = temp_img
            return self.return_img
        else:
            img_hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
            img_hsv[:,:,2] = convolve2d(img_hsv[:,:,2], kernel, mode="same")
            img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            self.return_img = img_rgb
            return self.return_img

    def median_filter_test(self, array, size):
        temp_img = array
        temp_img = Image.fromarray(temp_img)
        temp_img = temp_img.filter(ImageFilter.MedianFilter(size))
        self.return_img = np.array(temp_img)
        return self.return_img

    def non_linear_test(self, array):
        filter1 = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
        filter2 = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]])
        temp_img = array
        x = convolve2d(abs(temp_img), filter1)
        y = convolve2d(abs(temp_img), filter2)
        temp_img = (np.abs(x) + np.abs(y))
        self.return_img = temp_img
        return self.return_img

    def generic_filter_test(self, array ,matrix):
        generic = matrix
        temp_img = array
        temp_img = convolve2d(temp_img, generic, mode="same")
        self.return_img = temp_img
        return self.return_img

    def limiar_test(self, array, num):
        temp_img = array
        temp_img = temp_img >= num
        self.return_img = temp_img
        return self.return_img

    def fourier_test(self, array):
        temp_img = array
        temp_img = temp_img/255
        temp_img = np.fft.fft2(temp_img)
        temp_img = np.fft.fftshift(temp_img)
        temp_img = fc.normalize_img(np.absolute(temp_img).clip(0,1000))
        self.return_img = (np.uint8(np.clip(np.real(temp_img), 0, 255)))
        return (np.real(self.return_img))

    def high_fourier_test(self, array, radius=20):
        temp_img = array
        temp_img = temp_img/255
        temp_img = np.fft.fft2(temp_img)
        temp_img = np.fft.fftshift(temp_img)

        mask = fc.create_circular_mask(temp_img.shape[0], temp_img.shape[1], radius=radius)
        temp_img = mask * temp_img

        temp_img = np.fft.ifftshift(temp_img)
        temp_img = np.fft.ifft2(temp_img)

        self.return_img = (np.uint8(np.clip(np.real(temp_img) * 255, 0, 255)))
        return (np.real(self.return_img))

    def low_fourier_test(self, array, radius=20):
        temp_img = array
        temp_img = temp_img/255
        temp_img = np.fft.fft2(temp_img)
        temp_img = np.fft.fftshift(temp_img)

        mask = fc.create_circular_mask(temp_img.shape[0], temp_img.shape[1], radius=radius)
        temp_img = (1 - mask) * temp_img

        temp_img = np.fft.ifftshift(temp_img)
        temp_img = np.fft.ifft2(temp_img)

        self.return_img = (np.uint8(np.clip(np.real(temp_img) * 255, 0, 255)))
        return (np.real(self.return_img))

    def gray_scale_mean_test(self, array):
        temp_img = np.mean(array, axis=2)
        self.return_img = temp_img
        return self.return_img

    def gray_scale_avg_test(self, array):
        temp_img = np.average(array, weights=[0.299, 0.587, 0.114], axis=2)
        self.return_img = temp_img
        return self.return_img

    def saturation_enc_test(self, array, num):
        temp_img = array
        
        image = Image.fromarray(temp_img)
        img = image.convert('RGB')
        arr = np.array(np.asarray(img).astype('float'))
        self.return_img = fc.shift_saturation(arr, num).astype('uint8')

        # self.return_img = np.array((arr * 255)).astype(np.uint8)
        return self.return_img

    def hue_test(self, array, hue=270):
        temp_img = array
        image = Image.fromarray(temp_img)
        img = image.convert('RGBA')
        arr = np.array(np.asarray(img).astype('float'))
        self.return_img = fc.shift_hue(arr, hue/360.).astype('uint8')

        # self.img_now = np.array((arr * 255)).astype(np.uint8)
        return self.return_img

    def serpia_test(self, array):
        temp_img = array
        temp_img = temp_img/255
        sp = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]]).T
        temp_img = temp_img @ sp
        self.return_img = (temp_img * 255).clip(0,255).astype(np.uint8)
        return self.return_img

    def convert(self, array):
        try:
            return Image.fromarray(array)
        except:
            return fc.from_array(array)

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
                temp_img[i][j] = fc.linear_parts(
                    points_x, points_y, temp_img[i][j])
        temp_img = (temp_img * 255).astype(np.uint8)
        self.img_now = temp_img
        return Image.fromarray(self.img_now)

    def laplacian_filter_apply(self):
        laplacian = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])
        if len(self.img_now.shape) < 3:
            temp_img = self.img_now
            temp_img = convolve2d(abs(temp_img), laplacian)
            shp = self.img_now.shape
            ip = temp_img[0:shp[0], 0:shp[1]]
            self.img_now = self.img_now + (0.3*ip)
            return Image.fromarray(self.img_now)
        else:
            img_hsv = cv2.cvtColor(self.img_now, cv2.COLOR_RGB2HSV)
            img_hsv[:,:,2] = (convolve2d(img_hsv[:,:,2], laplacian, mode="same"))*0.05
            img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            self.img_now = (((self.img_now/255) + (img_rgb/255)*0.15)*255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(self.img_now)

    def high_boost_filter_apply(self):
        gaussian = np.array([[1, 2, 1],
                             [2, -8, 2],
                             [1, 2, 1]])
        if len(self.img_now.shape) < 3:
            temp_img = self.img_now
            shp = self.img_now.shape
            ipg = convolve2d(abs(temp_img), gaussian)
            ipg = ipg[0:shp[0], 0:shp[1]]
            ib = temp_img - ipg
            ib = ib[0:shp[0], 0:shp[1]]
            self.img_now = self.img_now + (0.05*ib)
            return Image.fromarray(self.img_now)
        else:
            img_hsv = cv2.cvtColor(self.img_now, cv2.COLOR_RGB2HSV)
            img_hsv[:,:,2] = (convolve2d(img_hsv[:,:,2], gaussian, mode="same"))*0.35
            img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            self.img_now = (((self.img_now/255) + (img_rgb/255)*0.1)*255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(self.img_now)

    def sobel_x_filter_apply(self):
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        temp_img = self.img_now
        temp_img = convolve2d(abs(temp_img), sobel_x)
        shp = self.img_now.shape
        ip = temp_img[0:shp[0], 0:shp[1]]
        self.img_now = self.img_now + (0.3*ip)
        return Image.fromarray(self.img_now)

    def sobel_y_filter_apply(self):
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        temp_img = self.img_now
        temp_img = convolve2d(abs(temp_img), sobel_y)
        shp = self.img_now.shape
        ip = temp_img[0:shp[0], 0:shp[1]]
        self.img_now = self.img_now + (0.3*ip)
        return Image.fromarray(self.img_now)

    def generic_filter(self, matrix):
        generic = matrix
        temp_img = self.img_now
        temp_img = convolve2d(temp_img, generic, mode="same")
        self.img_now = temp_img
        return Image.fromarray(self.img_now)

    def non_linear(self):
        filter1 = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
        filter2 = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]])
        temp_img = self.img_now
        x = convolve2d(abs(temp_img), filter1)
        y = convolve2d(abs(temp_img), filter2)
        temp_img = np.abs(x) + np.abs(y)
        shp = self.img_now.shape
        ip = temp_img[0:shp[0], 0:shp[1]]
        self.img_now = ip
        return Image.fromarray(self.img_now)

    def limiar(self):
        temp_img = self.img_now
        temp_img = temp_img >= 125
        self.img_now = temp_img
        return Image.fromarray(self.img_now)

    def fourier(self):
        temp_img = self.img_now
        temp_img = temp_img/255
        temp_img = np.fft.fft2(temp_img)
        temp_img = np.fft.fftshift(temp_img)
        temp_img = fc.normalize_img(np.absolute(temp_img).clip(0,1000))
        self.img_now = (np.uint8(np.clip(np.real(temp_img), 0, 255)))
        return fc.from_array(np.real(temp_img))

    def high_fourier(self, radius=20):
        temp_img = self.img_now
        temp_img = temp_img/255
        temp_img = np.fft.fft2(temp_img)
        temp_img = np.fft.fftshift(temp_img)

        mask = fc.create_circular_mask(temp_img.shape[0], temp_img.shape[1], radius=radius)
        temp_img = mask * temp_img

        temp_img = np.fft.ifftshift(temp_img)
        temp_img = np.fft.ifft2(temp_img)

        self.img_now = (np.uint8(np.clip(np.real(temp_img) * 255, 0, 255)))
        return fc.from_array(np.real(temp_img))

    def low_fourier(self, radius=20):
        temp_img = self.img_now
        temp_img = temp_img/255
        temp_img = np.fft.fft2(temp_img)
        temp_img = np.fft.fftshift(temp_img)

        mask = fc.create_circular_mask(temp_img.shape[0], temp_img.shape[1], radius=radius)
        temp_img = (1 - mask) * temp_img

        temp_img = np.fft.ifftshift(temp_img)
        temp_img = np.fft.ifft2(temp_img)

        self.img_now = (np.uint8(np.clip(np.real(temp_img) * 255, 0, 255)))
        return fc.from_array(np.real(temp_img))

    def gray_scale_mean(self):
        temp_img = np.mean(self.img_now, axis=2)
        self.img_now = temp_img
        return Image.fromarray(self.img_now)

    def gray_scale_avg(self):
        temp_img = np.average(self.img_now, weights=[0.299, 0.587, 0.114], axis=2)
        self.img_now = temp_img
        return Image.fromarray(self.img_now)

    def mean_simple_filter_apply(self, size):
        kernel = fc.generate_mean_simple_kernel(size)
        if len(self.img_now.shape) < 3:
            temp_img = self.img_now
            temp_img = convolve2d(abs(temp_img), kernel)
            self.img_now = temp_img
            return Image.fromarray(self.img_now)
        else:
            img_hsv = cv2.cvtColor(self.img_now, cv2.COLOR_RGB2HSV)
            img_hsv[:,:,2] = convolve2d(img_hsv[:,:,2], kernel, mode="same")
            img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            self.img_now = img_rgb
            return Image.fromarray(self.img_now)

    def mean_weighted_filter_apply(self, size):
        kernel = fc.generate_mean_weighted_kernel(size)
        if len(self.img_now.shape) < 3:
            temp_img = self.img_now
            temp_img = convolve2d(abs(temp_img), kernel)
            self.img_now = temp_img
            return Image.fromarray(self.img_now)
        else:
            img_hsv = cv2.cvtColor(self.img_now, cv2.COLOR_RGB2HSV)
            img_hsv[:,:,2] = convolve2d(img_hsv[:,:,2], kernel, mode="same")
            img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            self.img_now = img_rgb
            return Image.fromarray(self.img_now)

    def median_filter_apply(self, size):
        temp_img = self.img_now
        temp_img = Image.fromarray(temp_img)
        temp_img = temp_img.filter(ImageFilter.MedianFilter(size))
        self.img_now = np.array(temp_img)
        return Image.fromarray(self.img_now)

    def colored_hist(self):
        colors = ["red", "green", "blue"]
        channel_id = [0, 1, 2]

        plt.xlim([-10,266])
        for id, col in zip(channel_id, colors):
            histogram, bin_edges = np.histogram(self.img_now[:, :, id], bins=256, range=(0, 256))
            plt.plot(bin_edges[0:-1], histogram, color=col)

        plt.title('Colored Hist', size=15)
        plt.xlabel("Color value")
        plt.ylabel("Pixels")

        strfile = './media/colored_hist.png'
        plt.savefig(strfile, dpi=100)
        plt.close()
        return Image.open(strfile)

    def resize_i(self, fator=0.5):
        w1 = self.img_now.shape[0]
        h1 = self.img_now.shape[1]

        w2 = int(np.floor(w1*fator))
        h2 = int(np.floor(h1*fator))

        if len(self.img_now.shape) < 3:
            temp = np.empty((w2,h2))
        else:
            temp = np.empty((w2,h2,3))
        for j in range(h2):
            for i in range(w2):
                px = min(int(np.floor(i/fator)), w1-1)
                py = min(int(np.floor(j/fator)), h1-1)
                temp[i,j] = self.img_now[px,py]

        self.img_now = temp.astype(np.uint8)
        
        return Image.fromarray(self.img_now)

    def resize_i_bi(self, fator=0.5):
        w1 = self.img_now.shape[0]
        h1 = self.img_now.shape[1]

        w2 = int(np.floor(w1*fator))
        h2 = int(np.floor(h1*fator))

        if len(self.img_now.shape) < 3:
            temp = np.empty((w2,h2))
        else:
            temp = np.empty((w2,h2,3))
        for j in range(h2):
            for i in range(w2):
                scalerx = i/fator
                scalery = j/fator
                x = min(int(np.floor(i/fator)), w1-1)
                y = min(int(np.floor(j/fator)), h1-1)
                x2 = min(x+1, w1-1)
                y2 = min(y+1, h1-1)
                p1 = (x2-scalerx)*self.img_now[x,y]+(scalerx-x)*self.img_now[x2,y]
                p2 = (x2-scalerx)*self.img_now[x,y2]+(scalerx-x)*self.img_now[x2,y2]
                if x == x2:
                    p1 = self.img_now[x,y]
                    p2 = self.img_now[x2,y2]
                if y == y2:
                    p = self.img_now[x2,y2]
                else:
                    p = (y2-scalery)*p1+(scalery-y)*p2
                temp[i,j] = p

        self.img_now = temp.astype(np.uint8)
        
        return Image.fromarray(self.img_now)

    def rotate_i(self, rot=90):
        img_new = Image.fromarray(self.img_now)
        img_rot = img_new.rotate(rot, expand=True)
        self.img_now = np.array(img_rot)
        return Image.fromarray(self.img_now)

    def to_hsv(self):
        temp_img = self.img_now
        temp_img = temp_img/255
        temp_img = rgb2hsv(temp_img)
        self.img_now = temp_img
        self.img_now = (np.uint8(np.clip(np.real(temp_img) * 255, 0, 255)))
        return fc.from_array(self.img_now)

    def hue(self, hue=270):
        temp_img = self.img_now
        
        image = Image.fromarray(temp_img)
        img = image.convert('RGBA')
        arr = np.array(np.asarray(img).astype('float'))
        new_img = Image.fromarray(fc.shift_hue(arr, hue/360.).astype('uint8'), 'RGBA')

        self.img_now = np.array((arr * 255)).astype(np.uint8)
        return new_img

    def saturation(self, saturation=0.65):
        temp_img = self.img_now
        
        image = Image.fromarray(temp_img)
        img = image.convert('RGB')
        arr = np.array(np.asarray(img).astype('float'))
        new_img = Image.fromarray(fc.shift_saturation(arr, saturation).astype('uint8'), 'RGB')

        self.img_now = np.array((arr * 255)).astype(np.uint8)
        return new_img

    def serpia(self):
        temp_img = self.img_now
        temp_img = temp_img/255
        sp = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]]).T
        temp_img = temp_img @ sp
        self.img_now = (temp_img * 255).clip(0,255).astype(np.uint8)
        return Image.fromarray(self.img_now)

    def img_to_array(self):
        img_byte_arr = io.BytesIO()
        self.img.save(img_byte_arr, format=self.img.format)
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr

    def encrypt(self, text):

        img = cv2.cvtColor(np.asarray(self.img), cv2.COLOR_BGR2RGB)

        character_list = []

        for i in text:
            character_list.append(format(ord(i), '08b'))

        height, width, color = img.shape

        PixReq = len(character_list) * 3

        RowReq = PixReq/width
        RowReq = ceil(RowReq)

        width_parser = 0
        current_char_count = 0
        # Step 3
        for i in range(RowReq + 1):
            # Step 4
            while(width_parser < width and current_char_count < len(character_list)):
                binary_char = character_list[current_char_count]
                current_char_count += 1
                # Step 5
                for index_k, k in enumerate(binary_char):
                    if((k == '1' and img[i][width_parser][index_k % 3] % 2 == 0) or (k == '0' and img[i][width_parser][index_k % 3] % 2 == 1)):
                        img[i][width_parser][index_k % 3] -= 1
                    if(index_k % 3 == 2):
                        width_parser += 1
                    if(index_k == 7):
                        if(current_char_count*3 < PixReq and img[i][width_parser][2] % 2 == 1):
                            img[i][width_parser][2] -= 1
                        if(current_char_count*3 >= PixReq and img[i][width_parser][2] % 2 == 0):
                            img[i][width_parser][2] -= 1
                        width_parser += 1
            width_parser = 0
        # Step 6
        # Write the encrypted image into a new file
        cv2.imwrite("media/encrypted_image.png", img)

        return img

    def decrypt(self, img):
        character_list = []
        stop = False
        for index_i, i in enumerate(img):
            i.tolist()
            for index_j, j in enumerate(i):
                if((index_j) % 3 == 2):
                    # first pixel
                    character_list.append(bin(j[0])[-1])
                    # second pixel
                    character_list.append(bin(j[1])[-1])
                    # third pixel
                    if(bin(j[2])[-1] == '1'):
                        stop = True
                        break
                else:
                    # first pixel
                    character_list.append(bin(j[0])[-1])
                    # second pixel
                    character_list.append(bin(j[1])[-1])
                    # third pixel
                    character_list.append(bin(j[2])[-1])
            if(stop):
                break

        decoded_text = []
        # join all the bits to form letters (ASCII Representation)
        for i in range(int((len(character_list)+1)/8)):
            decoded_text.append(character_list[i*8:(i*8+8)])
        # join all the letters to form the decoded_text.
        decoded_text = [chr(int(''.join(i), 2)) for i in decoded_text]
        decoded_text = ''.join(decoded_text)

        return decoded_text

    def chroma_key(self, fln):
        temp_img = self.img_now
        mask = temp_img[:,:,1] <= 251
        mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
        
        temp_img = mask * temp_img 
        
        img_new = Image.open(fln)
        img_new = img_new.resize((temp_img.shape[1], temp_img.shape[0]))
        img_new = np.array(img_new).astype(np.uint8)

        self.img_now = (temp_img + (1- mask) * img_new).astype(np.uint8)
        return Image.fromarray(self.img_now)


if __name__ == "__main__":
    img = Image.open(
        '../../Downloads/pele_chroma.jpg')
    img.show()

    array = np.array(img)
    print(array.shape)
    verde = array[:,:,1]
    new_verde = np.empty((verde.shape[0], verde.shape[1]))
    for i, row in enumerate(verde):
        for j, element in enumerate(row):
            if element > 250:
                new_verde[i,j] = 0
            else:
                new_verde[i,j] = element

    array[:,:,1] = new_verde
    # array[:,:,1] = array[:,:,1]/255
    img_g = Image.fromarray(array)
    img_g.show()

    # img_insert = Image.open('../../Downloads/image.jpeg')
    # img_insert.show()

    # img_insert_array = np.array(img_insert)

    # for i, row in enumerate(img_insert_array):
    #     for j, element in enumerate(row):
    #         if array[i,j] > 50:
    #             img_insert_array[i,j] = array[i,j]

    # to_show = Image.fromarray(img_insert_array)
    # to_show.show()
