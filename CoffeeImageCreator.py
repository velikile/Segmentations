import cv2
import ImageSegment as IS
import file_utils as fu
import numpy as np
import random as rnd
import os
import argparse
from random import sample

IWIDTH =  3*224 
IHEIGHT = 3*224 
IMAGES_PER_SYNT_IMAGE = 10  
IMAGE_GEN_COUNT = 2

def CreateFolderForGeneratedImages(imageTrainDataPath = 'CoffeeBeanTrain/',imageValidationDataPath = 'CoffeeBeanValidation/'):
    if not os.path.exists(imageTrainDataPath):
        os.mkdir(imageTrainDataPath)
    if not os.path.exists(imageValidationDataPath):
        os.mkdir(imageValidationDataPath)
    return imageTrainDataPath,imageValidationDataPath

fileList = ["c1.jpeg","c2.jpeg"]

#finalImage = np.zeros((IWIDTH,IHEIGHT,3),np.uint8)
finalImage = 20 * np.ones((IWIDTH,IHEIGHT,3),np.uint8)
maskImage  = np.zeros((IWIDTH,IHEIGHT),np.uint8)

def CreateImageAndMask(image,thresh,bb,currentImage,currentMask,beanCounter): 
    minX,minY,maxX,maxY = bb.copy()
    beanImage = image[minX:maxX,minY:maxY]
    beanMask = thresh[minX:maxX,minY:maxY]
    w,h,c = image.shape
    wb,hb,cb = beanImage.shape
    xBean = np.random.randint(IWIDTH-wb,size=1)[0]
    yBean = np.random.randint(IHEIGHT-hb,size=1)[0]
    minX,maxX,minY,maxY = xBean,xBean+wb,yBean,yBean+hb
    beanMaskCond = beanImage > 0
    temp = currentImage[minX:maxX,minY:maxY,:]
    temp[beanMaskCond] = beanImage[beanMaskCond]
    currentImage[minX:maxX,minY:maxY,] = temp 
    temp = currentMask[minX:maxX,minY:maxY,]
    beanMaskCond = beanMask > 0
    temp[beanMaskCond] = beanCounter * beanMask[beanMaskCond]
    return currentImage,currentMask

def ShouldSkipOnBB(bb,r=range(45,55)):
    minX,minY,maxX,maxY = bb
    dx,dy = maxX - minX,maxY-minY 
    return dx not in r or dy not in r

def GetBBSForImages(fileList):
    fbbs = []
    images = []
    thresh_images = []
    for i,f in enumerate(fileList):
        image,thresh,contours = IS.GetContours(f)
        bbs = IS.GetBoundingBoxes(contours)
        tbbs =[]
        if(len(bbs) == 0):
            continue
        for bb in bbs :
            if ShouldSkipOnBB(bb):
                continue
            tbbs.append(bb)

        if(len(tbbs) > 0):
            fbbs.append(tbbs)
            images.append(image)
            thresh_images.append(thresh)

    return images,thresh_images,fbbs
    
def main(imageTrainDataPath,imageValidationDataPath):
    images,thresh_images,bbs = GetBBSForImages(fileList)
    for imageDataPath in [imageTrainDataPath,imageValidationDataPath] :
        for i in range(0,IMAGE_GEN_COUNT):
            count = 0 
            currentImage = finalImage.copy()
            currentMaskImage  = maskImage.copy()
            imageWritePath = imageDataPath + str(i) + 'i.png' 
            maskWritePath  = imageDataPath + str(i) + 'm.png' 
            for i,(image,thresh,bbsi) in enumerate(zip(images,thresh_images,bbs)):
                for beanCount,bb in enumerate(np.random.choice(len(bbsi),IMAGES_PER_SYNT_IMAGE)):
                    currentImage,currentMaskImage = CreateImageAndMask(image,thresh,bbsi[bb],currentImage,currentMaskImage,beanCount+1)
                    
                noiseScalar = np.random.randint(10,size=1)[0]
                noise = noiseScalar *  np.random.normal(0,1,currentImage.shape)

                cv2.imwrite(imageWritePath,cv2.blur(noise + currentImage,(3,3)))
                cv2.imwrite(maskWritePath,currentMaskImage)

def width_height_int(vstr):
    values = []
    vstrV = vstr.split(',')
    if(len(vstrV) != 2):
        return None
    for v in vstrV:
        values.append(int(v))
        
    return values

def InitParameters():
    global IWIDTH,IHEIGHT,IMAGES_PER_SYNT_IMAGE,IMAGE_GEN_COUNT,finalImage,maskImage
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--size',type=width_height_int)
    parser.add_argument('-c','--gen_count',type=int)
    parser.add_argument('-cpgi','--count_per_gen_image',type=int)
    args = parser.parse_args()
    if(args.size != None):
        IWIDTH,IHEIGHT = args.size
    if(args.count_per_gen_image != None):
        IMAGES_PER_SYNT_IMAGE = args.count_per_gen_image
    if(args.gen_count !=None):
        IMAGE_GEN_COUNT = args.gen_count
    finalImage = 20 * np.ones((IWIDTH,IHEIGHT,3),np.uint8)
    maskImage  = np.zeros((IWIDTH,IHEIGHT),np.uint8)

if __name__ == "__main__" :
    InitParameters()
    itd,ivd = CreateFolderForGeneratedImages()
    main(itd,ivd);
