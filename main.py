import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from img import Img

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

    img_test = img.convert(img_test)

    # atualiza a imagem na tela auxiliar
    img_s = ImageTk.PhotoImage(img_test)
    lbl_img.configure(image=img_s)
    lbl_img.image = img_s

# janela principal do programa
root = Tk()
root.title('GUI PDI')
root.geometry('650x900')
root.configure()

frm = Frame(root)
frm.pack(side=BOTTOM, padx=15, pady=15)
lbl = Label(root, text='Processamento de Imagens')
lbl.pack()

# botão de procurar os arquivos
btn_browse = Button(frm, text="Browse Image", command=select_image)
btn_browse.pack(side=tk.LEFT)

# botão de sair do programa
btn_exit = Button(frm, text="Exit", command=lambda: exit())
btn_exit.pack(side=tk.LEFT, padx=10)

# botão de aplicar as mudanças
frm2 = Frame(root)
frm2.pack(side=BOTTOM, padx=15, pady=15)
btn_apply = Button(frm2, text="Apply", command=apply)
btn_apply.pack(side=tk.LEFT, padx=10)

# scaler de brightness
frm3 = Frame(root)
frm3.pack(side=BOTTOM, padx=15, pady=15)
lbl_brightness = Label(frm3, text='Brightness')
lbl_brightness.pack(side=tk.TOP)
scl_brigh = Scale(frm3, from_=0, to=300, orient=HORIZONTAL, length=400, command=test)
scl_brigh.set(100)
scl_brigh.pack(side=tk.BOTTOM, padx=10)

# scaler de gama
frm4 = Frame(root)
frm4.pack(side=BOTTOM, padx=15, pady=15)
lbl_gama = Label(frm4, text='Gama')
lbl_gama.pack(side=tk.TOP)
scl_gama = Scale(frm4, from_=0, to=300, orient=HORIZONTAL, length=400, command=test)
scl_gama.set(100)
scl_gama.pack(side=tk.BOTTOM, padx=10)

# checkbox de negativar a imagem
frm5 = Frame(root)
frm5.pack(side=BOTTOM, padx=15, pady=15)
check_neg = IntVar()
box_negative = Checkbutton(frm5, text="Negative Image", variable=check_neg, command=test)
box_negative.pack(side=tk.TOP)


root.mainloop()
