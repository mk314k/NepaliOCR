import cv2
import numpy as np
from Rect import Rect
#import tensorflow as tf

FILEPATH = "D:\OCR_Project\images\\"
FILEPREFIX ="Gr9_Science_and_Technology_NP_CDC_1st_2079BS-page-"
REJECTWIDTHFRAC =0.7
REJECTHEIGHTFRAC =0.8
DETECTREGION =190

def contoursToBoundingRect(contours,imageWidth, imageHeight):
    result=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if w<REJECTWIDTHFRAC*imageWidth and h<REJECTHEIGHTFRAC*imageHeight:
            result.append(Rect(x,y,w,h))
    return result

def consecutiveRectMerge(rects):
    while True:
        i=0
        r=len(rects)
        while i<len(rects)-1:
            if rects[i].isOverlapping(rects[i+1]):
                rects[i]=rects[i]+rects[i+1]
                rects.remove(rects[i+1])
            i=i+1
        if len(rects)==r: break
    return rects

def fullRectMerge(rects):
    i=0
    r=len(rects)
    while i<r:
        j=0
        while j<r:
            if i!=j:
                if rects[i].isOverlapping(rects[j]) or rects[j].isOverlapping(rects[i]):
                    rects[i]=rects[i]+rects[j]
                    rects.remove(rects[j])
            j=j+1
            r=len(rects)
        i=i+1
        r=len(rects)
    return rects

def doubleContourDetect(img,imgWidth,imgHeight):
    for i in range(2):
        img_g =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        r, img_t=cv2.threshold(img_g,DETECTREGION,255,0)
        c,h=cv2.findContours(img_t,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if i==1: break
        cv2.drawContours(img,c,-1,(255,0,0),3)

    rects=contoursToBoundingRect(c,imgWidth,imgHeight)
    rectsMerged = consecutiveRectMerge(rects)
    rectsMerged = fullRectMerge(rects)

    return rectsMerged
   

def detectByBlur(filename:str):
    img = cv2.imread(FILEPATH+FILEPREFIX+filename)
    img_o=np.array(img)
    wid,hei,cha=img.shape

    img_g =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_b =cv2.GaussianBlur(img_g,(5,5),1  )
    r, img_t=cv2.threshold(img_b,DETECTREGION,255,0)
    c,h=cv2.findContours(img_t,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,c,-1,(255,0,0),3)
    cv2.imshow("contours only",img)

    # img_g =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # r, img_t=cv2.threshold(img_g,DETECTREGION,255,0)
    # c,h=cv2.findContours(img_t,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    rects=contoursToBoundingRect(c,wid,hei)
    rectsMerged = consecutiveRectMerge(rects)
    rectsMerged = fullRectMerge(rects)
    print(len(rectsMerged))

    for rect in rectsMerged:
        cv2.rectangle(img_o,(rect.x,rect.y),(rect.x+rect.width,rect.y+rect.height),(0,0,255),2)

    img_r = cv2.resize(img_o,(600,800))
    cv2.imshow("img14o",img_r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detector(filename:str,detectFunc=doubleContourDetect):
    img = cv2.imread(FILEPATH+FILEPREFIX+filename)
    img_o=np.array(img)

    # img_r = cv2.resize(img_o,(600,800))
    # cv2.imshow("img",img_r)

    imgWidth,imgHeight,imgChannel=img.shape
    rects=detectFunc(img,imgWidth,imgHeight)
    print(len(rects))

    for rect in rects:
        cv2.rectangle(img_o,(rect.x,rect.y),(rect.x+rect.width,rect.y+rect.height),(0,0,255),2)

    # img_r = cv2.resize(img_o,(600,800))
    # cv2.imshow("img_detcted",img_r)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return {
        "height":imgHeight,
        "width":imgWidth,
        "rects":rects
    }

#doubleContourDetect("024.jpg")
#detectByBlur("024.jpg")
detector("024.jpg")