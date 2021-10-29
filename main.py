import os
import sys
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from img import Img
import functions as fc

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

    # sistema de navegação de arquivos
    fln = filedialog.askopenfilename(initialdir=os.getcwd(
    ), title="Select Image File", filetypes=(("TIF files", "*.tif"), ("All Files", "*.*")))
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
    window.geometry(f'{size[0]}x{size[1]}')
    window.configure()
    lbl_img = Label(window)
    lbl_img.pack()
    lbl_img.configure(image=img_s)
    lbl_img.image = img_s

    # representação em bytes da imagem
    # img_byte_arr = img.img_to_array()

    # img_byte_arr = img_now

    # bytes_existentes = 0
    # for i, j in enumerate(img_byte_arr):
    #     bytes_existentes = i
    #     print(i, "->", j)
    
    # tbe_text = str(input("Digite o texto a ser encodado: "))
    # int_text = int.from_bytes(tbe_text.encode('utf-8'), byteorder=sys.byteorder)
    # bin_text = bin(int_text)

    # bytes_necessarios = int_text/16

    # print("Texto original: ", tbe_text)
    # print("Texto encodado em int: ", int_text)
    # print("Texto encodado em binario: ", bin_text)
    # print("Quantidade de bytes necessários para acomodar texto: ", bytes_necessarios)
    # print("Quantidade de bytes existentes: ", bytes_existentes)
    # if(bytes_existentes >= bytes_necessarios):
    #     print("Sim é possível")
    # else:
    #     print("Não é possível")


"""
Função que aplica as mundaças que foram selecionadas na interface principal
"""


def apply():
    # pega os valores informados nos scalers
    brightness = scl_brigh.get()/100
    gama = scl_gama.get()/100

    # realiza as transformações nas imagens de acordo com os valores
    img_now = img.brightness_apply(brightness)
    img_now = img.gama_apply(gama)
    if check_neg.get() == 1:
        img_now = img.negative_image()
    if check_log.get() == 1:
        img_now = img.log_apply()
    if check_equal.get() == 1:
        img_now = img.equalize_hist()
    if check_filter_laplacian.get() == 1:
        img_now = img.laplacian_filter_apply()
    if check_filter_mean_simple.get() == 1:
        size = option_kernel_size.get()
        img_now = img.mean_simple_filter_apply(size)
    if check_filter_mean_weighted.get() == 1:
        size = option_kernel_size.get()
        img_now = img.mean_weighted_filter_apply(size)
    if check_filter_median.get() == 1:
        size = option_kernel_size.get()
        img_now = img.median_filter_apply(size)
    
    if len(input_x.get()) > 0 and check_lp.get() == 1:
        points_x = input_x.get()
        points_y = input_y.get()
    
        points_x = fc.transform_points(points_x)
        points_y = fc.transform_points(points_y)

        img_now = img.linear_parts_apply(points_x, points_y)

    # atualiza a imagem na tela auxiliar
    img_s = ImageTk.PhotoImage(img_now)
    lbl_img.configure(image=img_s)
    lbl_img.image = img_s

    hist_plot = img.histogram_plot()
    img_hist_plot = ImageTk.PhotoImage(hist_plot)
    lbl_hist.configure(image=img_hist_plot)
    lbl_hist.image = img_hist_plot

    # volta para configurações iniciais
    scl_brigh.set(100)
    scl_gama.set(100)
    box_negative.deselect()
    box_log.deselect()
    box_equal_hits.deselect()
    box_lp.deselect()
    box_laplacian.deselect()
    box_mean_simple.deselect()
    box_mean_weighted.deselect()
    box_median.deselect()


"""
Função para alterar a imagem em tempo real na tela auxilar
"""


def test(*args):
    # pega os valores informados nos scalers
    brightness = scl_brigh.get()/100
    gama = scl_gama.get()/100

    # realiza as transformações nas imagens de acordo com os valores
    img_test = img.brightness_test(img.img_now, brightness)
    img_test = img.gama_test(img_test, gama)
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

def realizar_estegnografia():
    global img_encrypted
    img_encrypted = img.encrypt(entry_esteg.get())

def decodificar_estegnografia():
    frase_entrada.configure(text=f'O texto decodificado é: {img.decrypt(img_encrypted)}')


# janela principal do programa
root = Tk()
root.title('GUI PDI')
root.geometry('650x900')
root.configure()

curve = Toplevel(root)
curve.title("LP Plot")
curve.geometry('640x480')
curve.configure()
lbl_img_curve = Label(curve, text='Insert Values on Points X & Y and Press "See Plot"')
lbl_img_curve.pack()

filter = Toplevel(root)
filter.title("Extra")
filter.geometry('640x200')
filter.configure()

frm_filter = Frame(filter)
frm_filter.pack(side=BOTTOM, padx=15, pady=5)
check_filter_laplacian = IntVar()
box_laplacian = Checkbutton(frm_filter, text="Laplacian", variable=check_filter_laplacian, command=test)
box_laplacian.pack(side=tk.BOTTOM)

check_filter_high_boost = IntVar()
box_high_boost = Checkbutton(frm_filter, text="High Boost", variable=check_filter_high_boost, command=test)
box_high_boost.pack(side=tk.BOTTOM)

lbl_kernel = Label(frm_filter, text="Kernel Size: ")
lbl_kernel.pack(side=tk.LEFT)
OPTIONS = [3, 5, 7, 9, 11, 13, 15]
option_kernel_size = IntVar()
option_kernel_size.set(OPTIONS[0])
option_menu = OptionMenu(frm_filter, option_kernel_size, *OPTIONS, command=test)
option_menu.pack(side=tk.LEFT)

check_filter_mean_simple = IntVar()
box_mean_simple = Checkbutton(frm_filter, text="Mean Simple Filter", variable=check_filter_mean_simple, command=test)
box_mean_simple.pack(side=tk.RIGHT)

check_filter_mean_weighted = IntVar()
box_mean_weighted = Checkbutton(frm_filter, text="Mean Simple Weighted Filter", variable=check_filter_mean_weighted, command=test)
box_mean_weighted.pack(side=tk.RIGHT)

check_filter_median = IntVar()
box_median = Checkbutton(frm_filter, text="Median Filter", variable=check_filter_median, command=test)
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
box_equal_hits = Checkbutton(frm6, text="Equalize Histogram", variable=check_equal, command=test)
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
box_lp = Checkbutton(frm7, text="Linear Parts Apply", variable=check_lp, command=test) # depois testar com command=test
box_lp.pack(side=tk.BOTTOM)

frm8 = Frame(root)
frm8.pack(side=BOTTOM, padx=15, pady=5)
btn_esteg = Button(frm8, text="Decodificar Esteganografia", command=decodificar_estegnografia)
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
btn_esteg = Button(frm9, text="Gerar Esteganografia", command=realizar_estegnografia)
btn_esteg.pack(side=tk.LEFT, padx=10)


root.mainloop()
