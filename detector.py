import cv2
import numpy as np
from customDataStruct.RectSet import RectSet
from customDataStruct.Rect import Rect
from utility import *


def contourCorrection(imgo:np.ndarray,iteration=2)->np.ndarray:
    """AI is creating summary for contourCorrection

    Args:
        imgo (np.ndarray): [description]
        iteration (int, optional): [description]. Defaults to 2.

    Returns:
        np.ndarray: [description]
    """
    img = np.array(imgo)
    #TODO use cascading and remove this loop
    for i in range(iteration):
        imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #TODO use adaptive thresholding with suitable parameter instead of global thresholding used
        r, imgt = cv2.threshold(imgg,DETECTREGION,255,1)
        contours, hier = cv2.findContours(imgt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img,contours,-1,(255,0,0),i+2)#TODO contour color should be decided. A good approach is to draw inverse of image background
    return img, contours

def mergeAndClassify(contours:np.ndarray, img8:np.ndarray, padding=0, imgSource=None, rectsContainer = None, translateVector =(0,0))->RectSet:
    imgWidth,imgHeight,imgC =img8.shape
    if rectsContainer:
        extractedData = rectsContainer
    else:
        extractedData = RectSet((imgWidth,imgHeight),source=imgSource)
    for contour in contours:
        x,y,w,h= cv2.boundingRect(contour)
        newRect = Rect(x+translateVector[0],y+translateVector[1],width=w,height=h)
        if w<REJECTWIDTHFRAC*imgWidth and h<REJECTHEIGHTFRAC*imgHeight:
            cspr = colorSpread(img8,contour)
            if L1DistanceFrac((newRect.area(),cv2.contourArea(contour))) < AREAFACTOR:
                if cspr==0:
                    newRect.type = "Table"
                else:
                    newRect.type = 'Image'
            if newRect.type == 'text' and cspr >14:
                #TODO 14 works somehow but still to find proper reasoning
                cCnt, cSpr = colorCount(newRect,img8)
                if cCnt>12:
                    newRect.type='Image'   
            extractedData.addRect(newRect,padding)
    print('---------------------------------------All contour processed--------------------------')
    return extractedData

def doubleContourDetect(img_o:np.ndarray,padding=0,iteration=2,imgSource=None)->RectSet:
    img=np.array(img_o)
    img8=img//64*64+32 
    img8 = cv2.cvtColor(img8,cv2.COLOR_BGR2HSV)  
    img, contours = contourCorrection(img,iteration=iteration)
    print('----------------------------------all contours found------------------')
    #merging overlapped contours and classifying them as either text or table or image
    extractedData = mergeAndClassify(contours,img8,padding,imgSource)
    return extractedData

def contourRectDetect(img_o:np.ndarray,padding=0,iteration=2,imgSource=None, rectsContainer = None, translateVector = (0,0))->RectSet:
    img=np.array(img_o)
    img8=img//64*64+32 
    img8 = cv2.cvtColor(img8,cv2.COLOR_BGR2HSV)

    img, contours = contourCorrection(img,iteration=iteration)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),-1)
    img_g =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    r, img_t=cv2.threshold(img_g,DETECTREGION,255,1)
    contours,hier=cv2.findContours(img_t,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print('----------------------------------all contours found------------------')

    if rectsContainer:
        mergeAndClassify(contours,img8,padding,imgSource,rectsContainer=rectsContainer, translateVector=translateVector)
        return

    extractedData = mergeAndClassify(contours,img8,padding,imgSource)

    return extractedData
   

# def detectByBlur(imgo:np.ndarray):
#     img = np.array(imgo)
#     wid,hei,cha=img.shape
    
#     blurCx = 2*(wid//600)+1 
#     blurCy = 2*(hei//600)+1
#     kernel = np.ones((5,5),np.uint8)
#     imgd = cv2.dilate(img,kernel,iterations=1)
#     showImage("dilate",imgd)
#     imge = cv2.erode(imgd,kernel,iterations=1)
#     showImage("erode",imge)
#     img_g =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     showImage("gray",img_g)
#     #img_b =cv2.GaussianBlur(img,(blurCx,blurCy),25,25 )
#     img_b = cv2.medianBlur(img,min(blurCx,blurCy))
#     showImage("blurred",img_b)
#     detect(imgd,padding=20,iteration=1)
#     # r, img_t=cv2.threshold(img_b,DETECTREGION,255,0)
#     # c,h=cv2.findContours(img_t,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     # cv2.drawContours(img,c,-1,(255,0,0),3)
#     # cv2.imshow("contours only",img)

#     # for rect in rectsMerged:
#     #     print(rect)
#     #     cv2.rectangle(img_o,(rect.x,rect.y),(rect.x+rect.width,rect.y+rect.height),(0,0,255),2)

#     # showImage("img14o",img_r)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

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

def getAllLines(imgo:np.ndarray,maxDim = lambda img: img.shape[1]):
    maxdim = maxDim(imgo)
    imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    r, imgt = cv2.threshold(imgg,DETECTREGION,255,1)
    imgt=255-imgt
    
    allLines = np.count_nonzero(imgt,axis=1)

    peakLines =getPeaks(allLines,maxdim)
    return peakLines, imgt

def detectLines(img):
    imgHeight,imgWidth,imgChannel =img.shape
    peakLines,imgt =getAllLines(img)
    extractedRects = RectSet((imgHeight,imgWidth))

    for i in range(0,len(peakLines)-1,2):
        imgtt = imgt[peakLines[i]:peakLines[i+1],:]
        vlines = np.count_nonzero(imgtt,axis=0)
        vpeaks = getPeaks(vlines,peakLines[i+1]-peakLines[i])
        rect = Rect(vpeaks[0],peakLines[i],vpeaks[-1],peakLines[i+1])
        extractedRects.addRect(rect)
        
    return extractedRects

def detectBylines(img):
    imgHeight,imgWidth,imgChannel =img.shape
    peakLines,imgt =getAllLines(img)
    extractedRects = RectSet((imgHeight,imgWidth))

    for i in range(0,len(peakLines)-1,2):
        imgtt = imgt[peakLines[i]:peakLines[i+1],:]
        vlines = np.count_nonzero(imgtt,axis=0)
        vpeaks = getPeaks(vlines,peakLines[i+1]-peakLines[i])
        for j in range(0,len(vpeaks)-1,2):
            rect = Rect(vpeaks[j],peakLines[i],vpeaks[j+1],peakLines[i+1])
            extractedRects.addRect(rect)
        
    return extractedRects


def detect(imgo,padding=0,iteration=1)->RectSet:
    img=np.array(imgo)

    #Detect all the lines that spans horizontally/ heuristics for paragraph detection
    lines = detectLines(img)
    imgHeight,imgWidth,imgChannel =img.shape
    extractedRects = RectSet((imgHeight,imgWidth))
    #single contour correction and detection using countour Rect masking
    for rect in lines:
        origin = rect['upperLeft']
        end = rect['lowerRight']
        imgtt = img[end[1]:origin[1],origin[0]:end[0]]
        contourRectDetect(imgtt, iteration=iteration, padding=padding, rectsContainer= extractedRects, translateVector =(origin[0],end[1]))
        showImage('temp',imgtt)
        print(origin,end, rect)
    print(imgHeight,imgWidth)

    return extractedRects, lines

if __name__ == '__main__': 
    #img = cv2.imread("/Users/kartikeshmishra/Downloads/gr7scImg/img1.jpg")
    img = cv2.imread('/Users/kartikeshmishra/Kartikesh/NepaliOCR/NepaliOCR/trainImages/gr9sctrain24.jpg')
    #rects = doubleContourDetect(img,iteration=3)
    #rects = detectBylines(img)
    rects = doubleContourDetect(img)
    for rect in rects:
        colr = (255,0,0)
        if rect.type == 'Table':
            colr =(0,255,0)
        elif rect.type == 'text':
            colr = (0,0,255)
        cv2.rectangle(img,rect['lowerLeft'],rect['upperRight'],colr,3)
    # for rect in rects[0]:
    #     cv2.rectangle(img,rect['lowerLeft'],rect['upperRight'],(0,0,255),3)

    showImage("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
