import cv2
import numpy as np
import math

REJECTWIDTHFRAC =0.8
REJECTHEIGHTFRAC =0.8
DETECTREGION =190
AREAFACTOR =0.1
TABLEUNIQUECOLOR=5
COLORREDUCER=96
IMGMAXSIZE=1024

L1Distance = lambda p1p2: abs(p1p2[1]-p1p2[0])
L1DistanceFrac = lambda p1p2: L1Distance(p1p2)/p1p2[0]

def showImage(name,img):
    imgr=cv2.resize(img,(600,800))
    cv2.imshow(name,imgr)

def cascading(funcs, img,repeat=1):
    imgp=np.array(img)
    while repeat>0:
        for func in funcs:
            imgp=func(imgp)
        repeat-=1
    return imgp