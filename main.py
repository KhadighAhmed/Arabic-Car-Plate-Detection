import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import filedialog
import easyocr
from tkinter.filedialog import askopenfile
import sys
from PIL import Image, ImageTk
from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import imutils


app = tk.Tk()
app.geometry("500x300")
app.title("Egyptian License Plate Detector")
l1 = tk.Label(app, text='', width=50)
l1.pack()
b1 = tk.Button(app, text='Upload Image', width=20, height=2, command=lambda: plate_detection())
b1.pack(pady=20)
b2 = tk.Button(app, text="Exit", command=app.destroy)
b2.pack()
t = tk.Text(app, height=2, width=30)
t.pack(pady=20,padx=30, side=tk.LEFT)
b3 = tk.Button(app, text= "Clear", command=lambda:delete())
b3.pack(pady=20,padx=70, side=tk.RIGHT)

def delete():
   t.delete("1.0", "end")


def plate_detection():
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    input_image = cv2.imread(filename, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(input_image, (600, 400))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    plate_cascade = cv2.CascadeClassifier("haarcascade_license_plate.xml")
    plates = plate_cascade.detectMultiScale(blurred_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in plates:
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Original Image with Detected Plates", resized_image)
    cv2.waitKey(0)
    cv2.destroyWindow("Original Image with Detected Plates")
    for (x, y, w, h) in plates:
        plate_region = resized_image[y:y + h, x:x + w]

    cv2.imshow("Car's Plate", plate_region)
    cv2.waitKey(0)
    cv2.destroyWindow("Car's Plate")



    reader = easyocr.Reader(['ar'])
    result = reader.readtext(plate_region, detail=0)
    t.insert(tk.END, result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


app.mainloop()


