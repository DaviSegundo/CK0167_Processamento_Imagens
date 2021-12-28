import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import colorsys

def matriz_convert(n1: int, n2: int, n3: int, n4: int, n5: int, n6: int, n7: int, n8: int, n9: int):
    generic_filter = np.array([[n1,n2,n3],
                               [n4,n5,n6],
                               [n7,n8,n9]])
    return generic_filter

def shift_hue(arr, hout):
    rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = h + hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b))
    return arr

def shift_saturation(arr, sout):
    arr = arr/255
    rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    s = s * sout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b))
    arr = (arr * 255).clip(0,255).astype(np.uint8)
    return arr

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = np.uint8(dist_from_center <= radius)
    return mask

def from_array(image_data):
  return Image.fromarray(np.uint8(np.clip(image_data * 255, 0, 255)))

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

if __name__ == "__main__":
    n1,n2,n3,n4,n5,n6,n7,n8,n9 = 0,1,0,1,-4,1,0,1,0
    print(matriz_convert(n1,n2,n3,n4,n5,n6,n7,n8,n9))