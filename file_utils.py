import glob
from numpy import random as r
import numpy as np
import cv2 
import batch_create as bc

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


def GetImageBatch(path = 'images',width = 224,height = 224, channels = 3, fileExt = 'jpg',batchSize = 30 ,batchCount = 100):
    imageArr = GetImagesArray(path,width,height,channels,fileExt)
    batches = bc.CreateBatches(imageArr,batchSize,batchCount)
    return batches

def GetImageBatchNoP(path = 'images',fileExt = 'jpg',batchSize = 30 ,batchCount = 100):
    imageArr = GetImagesArrayNoP(path,fileExt)
    batches = bc.CreateBatches(imageArr,batchSize,batchCount)
    return batches

def GetImageBatchBaseAndMaskNoP(basePath = 'images',maskPath='images',fileExt = 'jpg',batchSize = 30 ,batchCount = 100):
    imageArr = GetImagesArrayNoP(basePath,fileExt)
    masksArr = GetImagesArrayNoP(maskPath,fileExt)
    imageMasksArr = []
    for i,j in zip(imageArr,masksArr):
        j = j[0:1,:,:]
        imageMasksArr.append([i,j])
    batches = bc.CreateBatches(imageMasksArr,batchSize,batchCount)

    return batches
