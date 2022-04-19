from msilib import Directory
import os
from unicodedata import category
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import pickle

Directory = r'C:\Users\Admin\OneDrive\Desktop\College\Tensorflow\dataset'
Category = ['Cardboard', 'Glass' , 'Metal', 'Paper', 'Plastic']
IMG_SIZE = 100
for category in Category:
    folder = os.path.join(Directory, category)
    label = Category.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        plt.imshow(img_arr)
        break