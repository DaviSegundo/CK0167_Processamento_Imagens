from PIL import Image
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import functions as fc
import cv2
from math import ceil


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
    
    def img_to_array(self):
        img_byte_arr = io.BytesIO()
        self.img.save(img_byte_arr, format=self.img.format)
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr

    def encrypt(data, imagem):
        # img = Image.convert(imagem, 'RGB')
        # img = np.array(img)
        img = cv2.cvtColor(np.asarray(imagem), cv2.COLOR_BGR2RGB)
        # imagem = Image(imagem).convert('RGB')
        # img_t = np.array(imagem)
        # img = img_t[:, :, ::-1].copy()

        # break the image into its character level. Represent the characyers in ASCII.
        data = [format(ord(i), '08b') for i in data]
        _, width, _ = img.shape
        # algorithm to encode the image
        PixReq = len(data) * 3

        RowReq = PixReq/width
        RowReq = ceil(RowReq)

        count = 0
        charCount = 0
        # Step 3
        for i in range(RowReq + 1):
            # Step 4
            while(count < width and charCount < len(data)):
                char = data[charCount]
                charCount += 1
                # Step 5
                for index_k, k in enumerate(char):
                    if((k == '1' and img[i][count][index_k % 3] % 2 == 0) or (k == '0' and img[i][count][index_k % 3] % 2 == 1)):
                        img[i][count][index_k % 3] -= 1
                    if(index_k % 3 == 2):
                        count += 1
                    if(index_k == 7):
                        if(charCount*3 < PixReq and img[i][count][2] % 2 == 1):
                            img[i][count][2] -= 1
                        if(charCount*3 >= PixReq and img[i][count][2] % 2 == 0):
                            img[i][count][2] -= 1
                        count += 1
            count = 0
        # Step 6
        # Write the encrypted image into a new file
        cv2.imshow("encrypted image", img)
        # cv2.imwrite("encrypted_image.png", img)
        return img
    
    def decrypt(img):

        data = []
        stop = False
        for index_i, i in enumerate(img):
            i.tolist()
            for index_j, j in enumerate(i):
                if((index_j) % 3 == 2):
                    # first pixel
                    data.append(bin(j[0])[-1])
                    # second pixel
                    data.append(bin(j[1])[-1])
                    # third pixel
                    if(bin(j[2])[-1] == '1'):
                        stop = True
                        break
                else:
                    # first pixel
                    data.append(bin(j[0])[-1])
                    # second pixel
                    data.append(bin(j[1])[-1])
                    # third pixel
                    data.append(bin(j[2])[-1])
            if(stop):
                break

        message = []
        # join all the bits to form letters (ASCII Representation)
        for i in range(int((len(data)+1)/8)):
            message.append(data[i*8:(i*8+8)])
        # join all the letters to form the message.
        message = [chr(int(''.join(i), 2)) for i in message]
        message = ''.join(message)

        print(message)