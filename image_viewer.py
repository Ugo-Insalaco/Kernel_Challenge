import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
import numpy as np

class ImageViewer():
    def __init__(self, images, labels, width=256, height=256):
        self.images = images
        self.labels = labels
        self.width = width
        self.height = height
        self.index = -1
        self.win = tk.Tk()

        self.btn_next = tk.Button(self.win,text='Next', command=self.next)
        self.btn_next.grid(row = 2, column = 0)
        self.btn_prev = tk.Button(self.win,text='Previous', command=self.previous)
        self.btn_prev.grid(row = 2, column = 1)

        self.img = tk.Label(self.win)
        self.img.grid(row=1,column=0)
        self.textbox = tk.Text(self.win,width=40,height=1)
        self.textbox.grid(row = 0, column = 0)

        self.next()
        self.win.protocol("WM_DELETE_WINDOW", self.destroy)


        self.win.mainloop()
    def destroy(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.win.destroy()
            self.win = None

    def update_image(self):
        img = 0.5*(self.images[self.index]+1)
        h, w = 32, 32
        img = np.reshape(img, (3, h, w))
        img = np.moveaxis(img, 0, 2) # N x h x w x 3
        img = Image.fromarray(np.uint8(img*255))
        img = img.resize((self.width,self.height), resample = Image.NEAREST)
        img = ImageTk.PhotoImage(img)
        self.img['image'] = None
        self.img.img = None
        self.img.img = img
        self.img['image'] = img
        self.textbox.delete("1.0","end")
        self.textbox.insert(tk.END,f'Class: {int(self.labels[self.index])}')

    def next(self):
        self.index+=1
        self.update_image()

    def previous(self):
        self.index-=1
        self.update_image()