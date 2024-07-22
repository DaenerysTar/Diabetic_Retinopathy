import numpy as np
import h5py
import os
from os import listdir
from os.path import isfile, join
import random
from scipy import misc
import matplotlib.pyplot as plt
import cv2
import multiprocessing
import copy

def scaleRadius(img, scale):
    x=img[int(img.shape[0]/2), :, :].sum(1)
    r=(x>x.mean()/10).sum()/2
    s=scale*1.0/r
    return cv2.resize(img, (0, 0), fx=s, fy=s)

scale = 300

def preprocess(mypath):
    a = cv2.imread(mypath)
    a = scaleRadius(a, scale)
    a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0),scale/30), -4, 128)
    b = np.zeros(a.shape)
    cv2.circle(b, (int(a.shape[1]/2), int(a.shape[0]/2)), int(scale*0.9), (1, 1, 1), -1, 8, 0)
    a = a*b+128*(1-b)
    #square crop
    side = min(a.shape[0:1])
    idx1 = int((a.shape[0]-side)/2)
    idx2 = int((a.shape[1]-side)/2)
    a = a[idx1:idx1+side, idx2:idx2+side, :]
    a_resized = np.empty((512, 512, 3))
    a_resized[:, :, 0] = cv2.resize(a[:, :, 0], (512, 512))
    a_resized[:, :, 1] = cv2.resize(a[:, :, 1], (512, 512))
    a_resized[:, :, 2] = cv2.resize(a[:, :, 2], (512, 512))
    #cv2.imwrite(newpath+img_name, a_resized)
    return a_resized
