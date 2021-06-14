import glob
from numpy import random as r
import numpy as np
import cv2 

def GetAllFiles(path,fileExt='png'):
    pathPattern = path+'/*.'+fileExt
    return glob.glob(pathPattern)

def GetImagesArray(path,width,height,channels,fileExt):
    files = GetAllFiles(path,fileExt)
    images = []
    for f in files :
        im = cv2.imread(f)
        im = cv2.resize(im,(width,height))
        if(channels == 3):
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        if(channels == 1):
            im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        images.append(im.reshape(channels,width,height))
        #images.append(im)
    return images

def GetImagesArrayNoP(path,fileExt):
    files = GetAllFiles(path,fileExt)
    images = []
    for f in files :
        im = cv2.imread(f)
        w,h,c = im.shape
        images.append(im.reshape((c,w,h)))

    return images
