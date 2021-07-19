import tkinter as tk
from PIL import Image as Pil_image, ImageTk as Pil_imageTk
from tkinter import filedialog
import numpy as np

from signModel import *


def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/home/anas/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    #print(image_data)
    basewidth = 700 
    img = Pil_image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Pil_image.ANTIALIAS)
    img = Pil_imageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()


def translate():
    z  = res(image_data)
    table = tk.Label(frame, text=z, font=("Arial",25)).pack()



root = tk.Tk()
root.title('Signboard Translator')
#root.iconbitmap('class.ico')
img = Pil_imageTk.PhotoImage(Pil_image.open('download.jpeg'))
root.tk.call('wm', 'iconphoto', root._w, img)
root.resizable(True, True)
tit = tk.Label(root, text="Signboard Translator", padx=25, pady=6, font=("", 12)).pack()
canvas = tk.Canvas(root, height=600, width=800, bg='grey')
canvas.pack()
frame = tk.Frame(root, bg='black')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
chose_image = tk.Button(root, text='Choose Image',
                        padx=35, pady=10,
                        fg="black", bg="white", command=load_img)
chose_image.pack(side=tk.LEFT)
class_image = tk.Button(root, text='Translate Image',
                        padx=35, pady=10,
                        fg="black", bg="white", command=translate)
class_image.pack(side=tk.RIGHT)
root.configure(bg='black')
root.mainloop()
