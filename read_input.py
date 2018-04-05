import tensorflow as tf
from function import *
import numpy as np
import skimage.measure

def read_input(path):
    image=np.load(path+"X.npy") #(3, 24, 10, 256, 513, 6)
    label=np.load(path+"Y.npy") #(3, 24, 10, 256, 513, 1)
    d,__,__,__,__,__ = np.shape(image)
    tr_image=image[0:int(d*0.8)]
    tr_label=label[0:int(d*0.8)]
    te_image=image[int(d*0.8):int(d*0.95)]
    te_label=label[int(d*0.8):int(d*0.95)]
    va_image=image[int(d*0.95):d]
    va_label=label[int(d*0.95):d]
    return tr_image,tr_label,te_image,te_label,va_image,va_label
