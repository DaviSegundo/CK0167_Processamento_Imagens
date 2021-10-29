import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def transform_points(text):
    list_s = text.split(",")

    points = []
    for i in list_s:
        points.append(float(i))
    
    return points

def linear_parts(points_x, points_y, x):
        n = len(points_x)

        sum = 0
        for i in range(n):
            prod = 1
            for j in range(n):
                if i != j:
                    prod *= ((x-points_x[j])/(points_x[i]-points_x[j]))
            sum += (points_y[i]) * prod
        sum = np.clip(sum, 0, 1)
        return sum

def plot_linear_parts(points_x, points_y):
    x_points = np.linspace(0, 1, 100)
    for i in x_points:
        plt.scatter(i, linear_parts(points_x, points_y, i), color='red')
    plt.title('Linear Parts Plot', size=15)

    strfile = './media/curve.png'
    plt.savefig(strfile, dpi=100)
    plt.close()
    return Image.open(strfile)

def generate_mean_simple_kernel(size):
    kernel = []
    for i in range(size):
        aux = []
        for j in range(size):
            aux.append(1)
        kernel.append(aux)
    kernel = np.array(kernel)
    kernel = kernel/(size**2)
    return kernel

def generate_mean_weighted_kernel(size):
    init = 1
    vals = []

    for i in range(size):
        if i == 0:
            vals.append(init)
        else:
            vals.append(init*2)
            init += 1

    kernel = []
    slide = (size//2)+1
    term = size//2

    var_aux = 0
    for i in range(slide):
        aux = []
        if i <= term:
            itens = vals[var_aux:slide+var_aux]
            var_aux += 1
        inv_itens = itens.copy()
        inv_itens.reverse()
        inv_itens.pop(0)
        itens.extend(inv_itens)
        kernel.append(itens)
    
    for i in range(term, 0, -1):
        kernel.append(kernel[i-1])

    pon = 0
    for k in kernel:
        pon += sum(k)

    kernel = np.array(kernel)
    kernel = kernel/pon

    return kernel

def normalize_img(array):
    norm = np.interp(array, (array.min(), array.max()), (0, 1))
    norm = norm*255
    return norm
