import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def GetMarkers(imagePath,distTransformThreshParam) :

    image = cv2.imread(imagePath)
    image = white_balance(image)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(255-gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)

    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations=1)
    sure_fg = cv2.dilate(thresh,kernel,iterations=2)
    dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,0)

    ret, sure_bg = cv2.threshold(dist_transform,distTransformThreshParam*dist_transform.max(),255,0)
    unknown = cv2.subtract(sure_fg,sure_bg)

    count, markers,stats,centroids = cv2.connectedComponentsWithStats(sure_fg,connectivity = 4)

    markers = markers+1

    markers[sure_bg == 255] = 0 
    
    image[sure_bg == 255] = 0

    image[unknown == 255] = 0

    markers = cv2.watershed(image ,markers)
    #markers = markers[1:,:-1]

    return image,markers

def GetContours(imagePath,distTransformThreshParam=0.2) :

    # returns the background as 0 and 1 for foreground
    image = cv2.imread(imagePath)
    image = white_balance(image)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(255-gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)

    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations=1)

    #sure_fg = cv2.dilate(thresh,kernel,iterations=2)
    #dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,0)

    #ret, sure_bg = cv2.threshold(dist_transform,distTransformThreshParam*dist_transform.max(),255,0)

    thresh[thresh == 255] = 1
    #thresh[sure_bg == 255] = 0
    image[thresh == 0] = 0

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return image,thresh,contours

def GetBoundingBoxes(contours):

    bbs = []
    for c in contours: 
        x = np.array([i[0][1] for i in c])
        y = np.array([i[0][0] for i in c])

        if(len(x) == 0 or len(y) == 0):
            continue 
        x = x.astype(int)
        y = y.astype(int)
        minX , maxX = np.min(x),np.max(x)
        minY , maxY = np.min(y),np.max(y)
        bbs.append([minX,minY,maxX,maxY])

    return bbs

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result
