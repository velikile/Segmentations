import cv2
import ImageSegment as IS
import file_utils as fu
import numpy as np
import random as rnd

IWIDTH =  720 
IHEIGHT = 720 

imageTrainDataPath = 'CoffeeBeanTrain/'
imageValidationDataPath = 'CoffeeBeanValidation/'
imageGenCount = 100 

#fileList = fu.GetAllFiles('images/base','jpg')
fileList = fu.GetAllFiles('../CoffeeBeans/images','jpeg')

#finalImage = np.zeros((IWIDTH,IHEIGHT,3),np.uint8)
finalImage = 20 * np.ones((IWIDTH,IHEIGHT,3),np.uint8)
maskImage  = np.zeros((IWIDTH,IHEIGHT),np.uint8)

def CreateImageAndMask(images,thresh,bb,currentImage,currentMask): 
    minX,minY,maxX,maxY = bb
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
    temp[beanMaskCond] = beanMask[beanMaskCond]
    return currentImage,currentMask

def ShouldSkipOnBB(bb):
    minX,minY,maxX,maxY = bb
    dx,dy = maxX - minX,maxY-minY 
    r = range(50,55)
    return dx not in r or dy not in r

for i in range(0,imageGenCount):
    count = 0 
    currentImage = finalImage.copy()
    currentMaskImage  = maskImage.copy()
    imageWritePath = imageTrainDataPath + str(i) + 'i.png' 
    maskWritePath = imageTrainDataPath + str(i) + 'm.png' 
    bbsWritePath = imageTrainDataPath + str(i) + 'bb.npy' 
    for f in fileList:
        count += 1 
        image,thresh,contours = IS.GetContours(f,10)
        bbs = IS.GetBoundingBoxes(contours)
        finalBBs = [] 
        for bb in bbs :
            if ShouldSkipOnBB(bb):
                continue

            finalBBs.append(bb)
            currentImage,currentMaskImage = CreateImageAndMask(image,thresh,bb,currentImage,currentMaskImage)
            #cv2.rectangle(maskImage,(minY,minX),(maxY,maxX),255,1)
        
        if(count == 10):
            
            noiseScalar = np.random.randint(10,size=1)[0]
            noise = noiseScalar *  np.random.normal(0,1,currentImage.shape)

            cv2.imwrite(imageWritePath,cv2.blur(noise + currentImage,(3,3)))
            cv2.waitKey()
            cv2.imwrite(maskWritePath,currentMaskImage)
            np.save(bbsWritePath,finalBBs)
            cv2.waitKey()
            count = 0 
            break

    rnd.shuffle(fileList)


