import cv2
import numpy as np
from RectSet import RectSet
from Rect import Rect
from utility import *

def colorSpread(img8, pointSet:np.ndarray):
    #TODO use dynamicNPy instead of list
    points =pointSet.flatten()
    colors = []
    n=len(points)
    for i in range(0,n,2):
        clr=img8[points[i+1]][points[i]][0]
        colors.append(clr)
    return np.std(np.array(colors))

def contourCorrection(imgo:np.ndarray,iteration=2)->np.ndarray:
    img = np.array(imgo)
    #TODO use cascading and remove this loop
    for i in range(iteration):
        imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        r, imgt = cv2.threshold(imgg,DETECTREGION,255,1)
        contours, hier = cv2.findContours(imgt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img,contours,-1,(255,0,0),i+2)
    return img, contours

def doubleContourDetect(img_o,padding=0,iteration=2,imgSource=None)->RectSet:
    img=np.array(img_o)
    img8=img//64*64+32 
    img8 = cv2.cvtColor(img8,cv2.COLOR_BGR2HSV)

    imgWidth,imgHeight,imgC =img.shape

    extractedData = RectSet((imgWidth,imgHeight),source=imgSource)
 
    
    for i in range(iteration):
        img_g =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img_t=cv2.adaptiveThreshold(img_g,255,cv2.ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,45,12) #TODO fix the parameters
        r, img_t=cv2.threshold(img_g,DETECTREGION,255,1)
        #showImage(f"thresh{i}",img_t)
        contours,hier=cv2.findContours(img_t,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #if i==1: break
        cv2.drawContours(img,contours,-1,(255,0,0),i+3)#TODO contour color should be decided. A good approach is to draw inverse of image background
        #showImage(f'img{i}',img)
    print('----------------------------------all contours found------------------')

    for contour in contours:
        x,y,w,h= cv2.boundingRect(contour)
        newRect = Rect(x,y,width=w,height=h)
        if w<REJECTWIDTHFRAC*imgWidth and h<REJECTHEIGHTFRAC*imgHeight:
            cspr = colorSpread(img8,contour)
            if L1DistanceFrac((newRect.area(),cv2.contourArea(contour))) <= AREAFACTOR:
                if cspr==0:
                    newRect.type = "Table"
                else:
                    newRect.type = 'Image'
            if newRect.type == 'text' and cspr >14:
                newRect.type='Image'   
            #newRect.type = 'Table'
            extractedData.addRect(newRect,padding)

    print('---------------------------------------All contour processed--------------------------')

    return extractedData,[]
   

def detectByBlur(imgo:np.ndarray):
    img = np.array(imgo)
    wid,hei,cha=img.shape
    
    blurCx = 2*(wid//600)+1 
    blurCy = 2*(hei//600)+1
    kernel = np.ones((5,5),np.uint8)
    imgd = cv2.dilate(img,kernel,iterations=1)
    showImage("dilate",imgd)
    imge = cv2.erode(imgd,kernel,iterations=1)
    showImage("erode",imge)
    img_g =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    showImage("gray",img_g)
    #img_b =cv2.GaussianBlur(img,(blurCx,blurCy),25,25 )
    img_b = cv2.medianBlur(img,min(blurCx,blurCy))
    showImage("blurred",img_b)
    detect(imgd,padding=20,iteration=1)
    # r, img_t=cv2.threshold(img_b,DETECTREGION,255,0)
    # c,h=cv2.findContours(img_t,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img,c,-1,(255,0,0),3)
    # cv2.imshow("contours only",img)

    # for rect in rectsMerged:
    #     print(rect)
    #     cv2.rectangle(img_o,(rect.x,rect.y),(rect.x+rect.width,rect.y+rect.height),(0,0,255),2)

    # showImage("img14o",img_r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect(img_o,detectFunc=doubleContourDetect,padding=0,iteration=2)->RectSet:
    img=np.array(img_o)
    allrects:RectSet=detectFunc(img,padding,iteration)

    def colorCount(rect:Rect)->int:
        return 1 
        p1=rect.lowerLeftPoint()
        p2=rect.upperRightPoint()
        imgd=imgc[p1[0]:p2[0],p1[1]:p2[1]]
        imgd=imgd.flatten()
        #imgd=np.reshape(imgd,(-1,3))
        u= np.unique(imgd,axis=0)
        return len(u)

    imgWidth,imgHeight,imgChannel=img.shape
    rects=allrects
    img8=img//128*128+64 
    img8 = cv2.cvtColor(img8,cv2.COLOR_BGR2HSV)

    for rect in rects:
        print(rect)
        colr = (0,0,255)
        if rect.type == 'Table': 
            colr = (0,255,0)
        elif rect.type =='Image':
            colr = (255,0,0)
        cv2.rectangle(img_o,(rect.lowerLeftPoint()[0],rect.lowerLeftPoint()[1]),(rect.upperRightPoint()[0],rect.upperRightPoint()[1]),colr,4)
    showImage("img_detcted",img_o)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return rects

def customDetector(rectDetection=None):
    return lambda img : rectDetection(img)

def getPeaks(allLines,maxLength):
    peakLines =[]
    prevI=0
    for lineI, lineW in enumerate(allLines):
        if lineW >= maxLength:
            if lineI - prevI>1:
                if prevI != -2 : peakLines.append(prevI)
                peakLines.append(lineI)
            prevI = lineI
    return peakLines

def detectBylines(img):
    imgHeight,imgWidth,imgChannel =img.shape
    imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    r, imgt = cv2.threshold(imgg,DETECTREGION,255,1)
    imgt=255-imgt
    
    allLines = np.count_nonzero(imgt,axis=1)

    peakLines =getPeaks(allLines,imgWidth)
    extractedRects = RectSet((imgHeight,imgWidth))
    allRects = []

    for i in range(0,len(peakLines)-1,2):
        imgtt = imgt[peakLines[i]:peakLines[i+1],:]
        vlines = np.count_nonzero(imgtt,axis=0)
        vpeaks = getPeaks(vlines,peakLines[i+1]-peakLines[i])
        for j in range(0,len(vpeaks)-1,2):
            rect = Rect(vpeaks[j],peakLines[i],vpeaks[j+1],peakLines[i+1])
            extractedRects.addRect(rect)
            allRects.append(rect)
        
    return extractedRects,allRects

if __name__ == '__main__': 
    img = cv2.imread("/Users/kartikeshmishra/Kartikesh/NepaliOCR/NepaliOCR/testImages/out23.jpg")
    rects = detectBylines(img)
    for rect in rects[1]:
        cv2.rectangle(img,rect['lowerLeft'],rect['upperRight'],(255,0,0),1)

    showImage("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
