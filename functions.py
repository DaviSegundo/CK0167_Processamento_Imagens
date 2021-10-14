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
