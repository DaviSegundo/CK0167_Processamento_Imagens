import os
import sys
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from img import Img
import functions as fc
from time import sleep

"""
Função que seleciona a imagem que vai ser trabalhada.
Cria uma janela auxilar para vizualização.
"""


def select_image():
    # variáveis globais
    global img
    global img_s
    global window
    global lbl_img
    global lbl_hist
    global curve
    global img_encrypted

    # sistema de navegação de arquivos
    fln = filedialog.askopenfilename(initialdir=os.getcwd(
    ), title="Select Image File", filetypes=(("All Files", "*.*"), ("TIF files", "*.tif")))
    img = Img(fln)
    size = img.size()
    img_s = ImageTk.PhotoImage(Image.open(fln))

    # plot inicial do histograma
    frm6 = Frame(root)
    frm6.pack(side=TOP, padx=15, pady=15)
    lbl_hist = Label(frm6)
    lbl_hist.pack(side=tk.TOP)

    hist_plot = img.histogram_plot()
    img_hist_plot = ImageTk.PhotoImage(hist_plot)
    lbl_hist.configure(image=img_hist_plot)
    lbl_hist.image = img_hist_plot

    # criação de uma nova janela para mostrar a imagem
    window = Toplevel(root)
    window.title("Imagem")
    # window.geometry(f'{size[0]}x{size[1]}')
    window.configure()
    lbl_img = Label(window)
    lbl_img.pack()
    lbl_img.configure(image=img_s)
    lbl_img.image = img_s


"""
Função que aplica as mundaças que foram selecionadas na interface principal
"""


def apply():
    # pega os valores informados nos scalers
    brightness = scl_brigh.get()/100
    gama = scl_gama.get()/100
    hue = scl_hue.get()
    saturation = scl_saturation.get()/100
    res = scl_resize.get()/100

    # realiza as transformações nas imagens de acordo com os valores
    img.img_now = img.img_original

    img_now = img.brightness_apply(brightness)
    img_now = img.gama_apply(gama)
    
    if check_chroma.get() == 1:
        img_now = img.chroma_key(fln_cho)
    if check_grayscale_mean.get() == 1:
        img_now = img.gray_scale_mean()
    if check_grayscale_avg.get() == 1:
        img_now = img.gray_scale_avg()
    if check_resize.get() == 1:
        img_now = img.resize_i(fator=res)
    if check_resize_b.get() == 1:
        img_now = img.resize_i_bi(fator=res)
    if check_neg.get() == 1:
        img_now = img.negative_image()
    if check_log.get() == 1:
        img_now = img.log_apply()
    if check_equal.get() == 1:
        img_now = img.equalize_hist()
    if check_filter_laplacian.get() == 1:
        img_now = img.laplacian_filter_apply()
    if check_filter_high_boost.get() == 1:
        img_now = img.high_boost_filter_apply()
    if check_filter_mean_simple.get() == 1:
        size = option_kernel_size.get()
        img_now = img.mean_simple_filter_apply(size)
    if check_filter_mean_weighted.get() == 1:
        size = option_kernel_size.get()
        img_now = img.mean_weighted_filter_apply(size)
    if check_filter_median.get() == 1:
        size = option_kernel_size.get()
        img_now = img.median_filter_apply(size)
    if check_filter_sobel_x.get() == 1:
        img_now = img.sobel_x_filter_apply()
    if check_filter_sobel_y.get() == 1:
        img_now = img.sobel_y_filter_apply()
    if check_filter_edge.get() == 1:
        img_now = img.non_linear()
    if check_high_fourier.get() == 1:
        radius = option_radius_size.get()
        img_now = img.high_fourier(radius=radius)
    if check_low_fourier.get() == 1:
        radius = option_radius_size.get()
        img_now = img.low_fourier(radius=radius)
    if check_fourier.get() == 1:
        img_now = img.fourier()
    if check_limiar.get() == 1:
        img_now = img.limiar()
    if check_hue.get() == 1:
        img_now = img.hue(hue=hue)
    if check_sat.get() == 1:
        img_now = img.saturation(saturation)
    if check_generic.get() == 1:
        img_now = img.generic_filter(fc.matriz_convert(-1,-1,-1,0,0,0,1,1,1))
    if check_rotate.get() == 1:
        rot = option_rotate_size.get()
        img_now = img.rotate_i(rot)
    if check_serpia.get() == 1:
        img_now = img.serpia()
    
    # img_now = img.saturation(saturation)

    if len(input_x.get()) > 0 and check_lp.get() == 1:
        points_x = input_x.get()
        points_y = input_y.get()

        points_x = fc.transform_points(points_x)
        points_y = fc.transform_points(points_y)

        img_now = img.linear_parts_apply(points_x, points_y)

    # window.geometry(f'{img_now.size[0]}x{img_now.size[1]}')

    # atualiza a imagem na tela auxiliar
    img_s = ImageTk.PhotoImage(img_now)
    lbl_img.configure(image=img_s)
    lbl_img.image = img_s

    hist_plot = img.histogram_plot()
    img_hist_plot = ImageTk.PhotoImage(hist_plot)
    lbl_hist.configure(image=img_hist_plot)
    lbl_hist.image = img_hist_plot

    # volta para configurações iniciais

    # scl_brigh.set(100)
    # scl_gama.set(100)
    # box_negative.deselect()
    # box_log.deselect()
    # box_equal_hits.deselect()
    # box_lp.deselect()
    # box_laplacian.deselect()
    # box_mean_simple.deselect()
    # box_mean_weighted.deselect()
    # box_median.deselect()
    # box_high_boost.deselect()
    # box_sobel_x.deselect()
    # box_sobel_y.deselect()
    # box_edge.deselect()
    # box_grayscale_mean.deselect()
    # box_grayscale_avg.deselect()
    # box_low_fourier.deselect()
    # box_high_fourier.deselect()
    # box_fourier.deselect()
    # box_limiar.deselect()
    # box_hsv.deselect()


"""
Função para alterar a imagem em tempo real na tela auxilar
"""


def test(*args):
    # pega os valores informados nos scalers
    brightness = scl_brigh.get()/100
    gama = scl_gama.get()/100
    hue = scl_hue.get()
    saturation = scl_saturation.get()/100
    limiar = scl_limiar.get()
    res = scl_resize.get()/100

    # realiza as transformações nas imagens de acordo com os valores
    img_test = img.brightness_test(img.img_now, brightness)
    img_test = img.gama_test(img_test, gama)

    if check_grayscale_mean.get() == 1:
        img_test = img.gray_scale_mean_test(img_test)
    if check_grayscale_avg.get() == 1:
        img_test = img.gray_scale_avg_test(img_test)
    if check_neg.get() == 1:
        img_test = img.negative_image_test(img_test)
    if check_log.get() == 1:
        img_test = img.log_test(img_test)
    if check_equal.get() == 1:
        img_test = img.equalize_hist_test(img_test)
    if check_filter_laplacian.get() == 1:
        img_test = img.laplacian_filter_test(img_test)
    if check_filter_mean_simple.get() == 1:
        size = option_kernel_size.get()
        img_test = img.mean_simple_filter_test(img_test, size)
    if check_filter_mean_weighted.get() == 1:
        size = option_kernel_size.get()
        img_test = img.mean_weighted_filter_test(img_test, size)
    if check_filter_median.get() == 1:
        size = option_kernel_size.get()
        img_test = img.median_filter_test(img_test, size)
    if check_filter_high_boost.get() == 1:
        img_test = img.high_boost_filter_test(img_test)
    if check_filter_sobel_x.get() == 1:
        img_test = img.sobel_x_filter_test(img_test)
    if check_filter_sobel_y.get() == 1:
        img_test = img.sobel_y_filter_test(img_test)
    if check_filter_edge.get() == 1:
        img_test = img.non_linear_test(img_test)
    if check_limiar.get() == 1:
        img_test = img.limiar_test(img_test, limiar)
    if check_fourier.get() == 1:
        img_test = img.fourier_test(img_test)
    if check_high_fourier.get() == 1:
        radius = option_radius_size.get()
        img_test = img.high_fourier_test(img_test, radius=radius)
    if check_low_fourier.get() == 1:
        radius = option_radius_size.get()
        img_test = img.low_fourier_test(img_test, radius=radius)
    if check_generic.get() == 1:
        img_test = img.generic_filter_test(img_test, fc.matriz_convert(0,1,0,1,-4,1,0,1,0))
    if check_sat.get() == 1:
        img_test = img.saturation_enc_test(img_test, saturation)
    if check_hue.get() == 1:
        img_test = img.hue_test(img_test, hue)
    if check_serpia.get() == 1:
        img_test = img.serpia_test(img_test)

    if len(input_x.get()) > 0 and check_lp.get() == 1:
        points_x = input_x.get()
        points_y = input_y.get()

        points_x = fc.transform_points(points_x)
        points_y = fc.transform_points(points_y)

        img_test = img.linear_parts_test(points_x, points_y, img_test)

    img_test = img.convert(img_test)

    # atualiza a imagem na tela auxiliar
    img_s = ImageTk.PhotoImage(img_test)
    lbl_img.configure(image=img_s)
    lbl_img.image = img_s


def show_linear():
    points_x = input_x.get()
    points_y = input_y.get()

    points_x = fc.transform_points(points_x)
    points_y = fc.transform_points(points_y)

    img_curve = fc.plot_linear_parts(points_x, points_y)
    img_curve = ImageTk.PhotoImage(img_curve)

    lbl_img_curve.configure(image=img_curve)
    lbl_img_curve.image = img_curve

def show_colored_hist():
    img_colored = img.colored_hist()
    img_colored = ImageTk.PhotoImage(img_colored)
    
    lbl_colored_hist.configure(image=img_colored)
    lbl_colored_hist.image = img_colored

def chroma_choice():
    global fln_cho
    fln_cho = filedialog.askopenfilename(initialdir=os.getcwd(
    ), title="Select Image File", filetypes=(("All Files", "*.*"), ("TIF files", "*.tif")))

def save():
    imga = Image.fromarray(img.img_now)
    imga.save("./media/saved_file.png")

def realizar_estegnografia():
    img_encrypted = img.encrypt(entry_esteg.get())


def decodificar_estegnografia():
    frase_entrada.configure(
        text=f'O texto decodificado é: {img.decrypt(img.img_original)}')


# janela principal do programa
root = Tk()
root.title('GUI PDI')
root.geometry('1850x950')
root.configure()

frm_side_side = Frame(root)
frm_side_side.pack(side=RIGHT, padx=15)

lbl_colored_hist = Label(frm_side_side)
lbl_colored_hist.pack(side=TOP)

frm_side = Frame(root)
frm_side.pack(side=RIGHT, padx=15)

frm_filter = Frame(frm_side)
frm_filter.pack(side=BOTTOM, padx=15)

lbl_img_curve = Label(frm_side_side)
lbl_img_curve.pack(side=TOP)

btn_col_plot = Button(frm_filter, text="See Colored Hist", command=show_colored_hist)
btn_col_plot.pack(side=tk.BOTTOM, pady=5)

frm_chroma = Frame(frm_filter)
frm_chroma.pack(side=BOTTOM, pady=5)

check_chroma = IntVar()
box_chroma = Checkbutton(frm_chroma, text="Chroma Key", variable=check_chroma)#, command=test)
box_chroma.pack(side=tk.RIGHT)
btn_cho = Button(frm_chroma, text="Background Image", command=chroma_choice)
btn_cho.pack(side=RIGHT)

frm_rotate = Frame(frm_filter)
frm_rotate.pack(side=BOTTOM)

check_rotate = IntVar()
box_rotate = Checkbutton(frm_rotate, text="Rotate", variable=check_rotate)
box_rotate.pack(side=tk.RIGHT)

OPTIONS_ROT = [90,180,270]
option_rotate_size = IntVar()
option_rotate_size.set(OPTIONS_ROT[0])
option_menu_radius = OptionMenu(
    frm_rotate, option_rotate_size, *OPTIONS_ROT)
option_menu_radius.pack(side=tk.RIGHT)

lbl_rotate = Label(frm_rotate, text="Degree: ")
lbl_rotate.pack(side=RIGHT)

frm_resize = Frame(frm_filter)
frm_resize.pack(side=BOTTOM)

check_resize_b = IntVar()
box_resize_b = Checkbutton(frm_resize, text="Resize BI", variable=check_resize_b)
box_resize_b.pack(side=tk.RIGHT)
check_resize = IntVar()
box_resize = Checkbutton(frm_resize, text="Resize NN", variable=check_resize)
box_resize.pack(side=tk.RIGHT)
scl_resize = Scale(frm_resize, from_=1, to=200, orient=HORIZONTAL, length=100)
scl_resize.set(100)
scl_resize.pack(side=RIGHT, padx=10)
lbl_resize = Label(frm_resize, text="Fator: ")
lbl_resize.pack(side=RIGHT)

scl_saturation = Scale(frm_filter, from_=10, to=200, orient=HORIZONTAL, length=300, command=test)
scl_saturation.set(0)
scl_saturation.pack(side=BOTTOM, padx=10)
check_sat = IntVar()
box_sat = Checkbutton(frm_filter, text="Saturation Confirm", variable=check_sat)
box_sat.pack(side=tk.BOTTOM)

scl_hue = Scale(frm_filter, from_=0, to=360, orient=HORIZONTAL, length=300, command=test)
scl_hue.set(0)
scl_hue.pack(side=BOTTOM, padx=10)
check_hue = IntVar()
box_hue = Checkbutton(frm_filter, text="Hue Confirm", variable=check_hue)
box_hue.pack(side=tk.BOTTOM)

check_serpia = IntVar()
box_serpia = Checkbutton(frm_filter, text="Serpia", variable=check_serpia, command=test)
box_serpia.pack(side=tk.BOTTOM)

check_grayscale_avg = IntVar()
box_grayscale_avg = Checkbutton(frm_filter, text="Gray Scale AVG", variable=check_grayscale_avg, command=test)
box_grayscale_avg.pack(side=tk.BOTTOM)

check_grayscale_mean = IntVar()
box_grayscale_mean = Checkbutton(frm_filter, text="Gray Scale Mean", variable=check_grayscale_mean, command=test)
box_grayscale_mean.pack(side=tk.BOTTOM)

lbl_saturation_hue = Label(frm_filter, text='Color Image')
lbl_saturation_hue.pack(side=BOTTOM, pady=5)

frm_fourier = Frame(frm_filter)
frm_fourier.pack(side=BOTTOM)

check_fourier = IntVar()
box_fourier = Checkbutton(frm_fourier, text="Des Fourier", variable=check_fourier, command=test)
box_fourier.pack(side=tk.RIGHT)

check_high_fourier = IntVar()
box_high_fourier = Checkbutton(frm_fourier, text="Low Fourier", variable=check_high_fourier, command=test)
box_high_fourier.pack(side=tk.RIGHT)

check_low_fourier = IntVar()
box_low_fourier = Checkbutton(frm_fourier, text="High Fourier", variable=check_low_fourier, command=test)
box_low_fourier.pack(side=tk.RIGHT)

# Insert Radius value here
OPTIONS_R = [10, 20, 30, 50, 75, 100, 150]
option_radius_size = IntVar()
option_radius_size.set(OPTIONS_R[0])
option_menu_radius = OptionMenu(
    frm_fourier, option_radius_size, *OPTIONS_R, command=test)
option_menu_radius.pack(side=tk.RIGHT)

label_radius = Label(frm_fourier, text="Radius: ")
label_radius.pack(side=RIGHT)

label_fourier = Label(frm_filter, text="Fourier")
label_fourier.pack(side=BOTTOM, pady=5)

check_filter_edge = IntVar()
box_edge = Checkbutton(
    frm_filter, text="Non Linear Edge Detection", variable=check_filter_edge, command=test)
box_edge.pack(side=tk.BOTTOM)

check_filter_sobel_y = IntVar()
box_sobel_y = Checkbutton(
    frm_filter, text="Sobel Y", variable=check_filter_sobel_y, command=test)
box_sobel_y.pack(side=tk.BOTTOM)
check_filter_sobel_x = IntVar()
box_sobel_x = Checkbutton(
    frm_filter, text="Sobel X", variable=check_filter_sobel_x, command=test)
box_sobel_x.pack(side=tk.BOTTOM)

check_filter_laplacian = IntVar()
box_laplacian = Checkbutton(
    frm_filter, text="Laplacian", variable=check_filter_laplacian, command=test)
box_laplacian.pack(side=tk.BOTTOM)

check_filter_high_boost = IntVar()
box_high_boost = Checkbutton(
    frm_filter, text="High Boost", variable=check_filter_high_boost, command=test)
box_high_boost.pack(side=tk.BOTTOM)

check_generic = IntVar()
box_generic = Checkbutton(frm_filter, text="Generic", variable=check_generic, command=test)
box_generic.pack(side=tk.BOTTOM)

frm_limiar = Frame(frm_filter)
frm_limiar.pack(side=BOTTOM)

check_limiar = IntVar()
box_limiar = Checkbutton(frm_limiar, text="Limiar", variable=check_limiar, command=test)
box_limiar.pack(side=tk.RIGHT)
scl_limiar = Scale(frm_limiar, from_=0, to=256, orient=HORIZONTAL, length=150, command=test)
scl_limiar.set(125)
scl_limiar.pack(side=RIGHT, padx=10)
lbl_limiar = Label(frm_limiar, text="Value: ")
lbl_limiar.pack(side=RIGHT)

lbl_filter_info = Label(frm_filter, text="Filters")
lbl_filter_info.pack(side=tk.TOP)

lbl_kernel = Label(frm_filter, text="Kernel Size: ")
lbl_kernel.pack(side=tk.LEFT)
OPTIONS = [3, 5, 7, 9, 11, 13, 15]
option_kernel_size = IntVar()
option_kernel_size.set(OPTIONS[0])
option_menu = OptionMenu(
    frm_filter, option_kernel_size, *OPTIONS, command=test)
option_menu.pack(side=tk.LEFT)

check_filter_mean_simple = IntVar()
box_mean_simple = Checkbutton(
    frm_filter, text="Mean Simple Filter", variable=check_filter_mean_simple, command=test)
box_mean_simple.pack(side=tk.RIGHT)

check_filter_mean_weighted = IntVar()
box_mean_weighted = Checkbutton(
    frm_filter, text="Mean Simple Weighted Filter", variable=check_filter_mean_weighted, command=test)
box_mean_weighted.pack(side=tk.RIGHT)

check_filter_median = IntVar()
box_median = Checkbutton(frm_filter, text="Median Filter",
                         variable=check_filter_median, command=test)
box_median.pack(side=tk.RIGHT)



"""
Tela Principal
"""
frm = Frame(root)
frm.pack(side=BOTTOM, padx=15, pady=15)

# botão de procurar os arquivos
btn_browse = Button(frm, text="Browse Image", command=select_image)
btn_browse.pack(side=tk.LEFT)

# botão de sair do programa
btn_exit = Button(frm, text="Exit", command=lambda: exit())
btn_exit.pack(side=tk.LEFT, padx=10)

# botão de aplicar as mudanças
frm2 = Frame(root)
frm2.pack(side=BOTTOM, padx=15, pady=10)
btn_apply = Button(frm2, text="Apply", command=apply)
btn_apply.pack(side=tk.LEFT, padx=10)
btn_save = Button(frm2, text="Save", command=save)
btn_save.pack(side=tk.LEFT, padx=10)

# scaler de brightness
frm3 = Frame(root)
frm3.pack(side=BOTTOM, padx=15, pady=5)
lbl_brightness = Label(frm3, text='Brightness')
lbl_brightness.pack(side=tk.TOP)
scl_brigh = Scale(frm3, from_=0, to=300, orient=HORIZONTAL,
                  length=400, command=test)
scl_brigh.set(100)
scl_brigh.pack(side=tk.BOTTOM, padx=10)

# scaler de gama
frm4 = Frame(root)
frm4.pack(side=BOTTOM, padx=15, pady=5)
lbl_gama = Label(frm4, text='Gama')
lbl_gama.pack(side=tk.TOP)
scl_gama = Scale(frm4, from_=0, to=300, orient=HORIZONTAL,
                 length=400, command=test)
scl_gama.set(100)
scl_gama.pack(side=tk.BOTTOM, padx=10)

# checkbox de negativar a imagem
frm5 = Frame(root)
frm5.pack(side=BOTTOM, padx=15, pady=5)
check_neg = IntVar()
box_negative = Checkbutton(frm5, text="Negative Image",
                           variable=check_neg, command=test)
box_negative.pack(side=tk.LEFT)

check_log = IntVar()
box_log = Checkbutton(frm5, text="Log2", variable=check_log, command=test)
box_log.pack(side=tk.RIGHT)

frm6 = Frame(root)
frm6.pack(side=BOTTOM, padx=15, pady=5)
check_equal = IntVar()
box_equal_hits = Checkbutton(
    frm6, text="Equalize Histogram", variable=check_equal, command=test)
box_equal_hits.pack(side=tk.LEFT)

frm7 = Frame(root)
frm7.pack(side=BOTTOM, padx=15, pady=5)
lbl_lp = Label(frm7, text='Linear Parts Points')
lbl_lp.pack(side=tk.TOP)
text_x = Label(frm7, text='Pontos X:')
text_x.pack(side=tk.LEFT)
input_x = Entry(frm7)
input_x.pack(side=tk.LEFT)
text_y = Label(frm7, text='Pontos Y:')
text_y.pack(side=tk.LEFT)
input_y = Entry(frm7)
input_y.pack(side=tk.LEFT)
btn_plot = Button(frm7, text="See Plot", command=show_linear)
btn_plot.pack(side=tk.LEFT, padx=10)
check_lp = IntVar()
box_lp = Checkbutton(frm7, text="Linear Parts Apply", variable=check_lp,
                     command=test)  # depois testar com command=test
box_lp.pack(side=tk.BOTTOM)

frm8 = Frame(root)
frm8.pack(side=BOTTOM, padx=15, pady=5)
btn_esteg = Button(frm8, text="Decodificar Esteganografia",
                   command=decodificar_estegnografia)
btn_esteg.pack(side=tk.TOP, padx=10)
frase_entrada = Label(frm8, text='')
frase_entrada.pack(side=tk.BOTTOM)

frm9 = Frame(root)
frm9.pack(side=BOTTOM, padx=15, pady=5)
lbl_esteg = Label(frm9, text='Esteganografia')
lbl_esteg.pack(side=tk.TOP)
lbl_esteg = Label(frm9, text='Entre uma frase: ')
lbl_esteg.pack(side=tk.LEFT)
entry_esteg = Entry(frm9)
entry_esteg.pack(side=tk.LEFT)
btn_esteg = Button(frm9, text="Gerar Esteganografia",
                   command=realizar_estegnografia)
btn_esteg.pack(side=tk.LEFT, padx=10)


root.mainloop()
