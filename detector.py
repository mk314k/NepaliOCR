import cv2
from cv2 import THRESH_BINARY
import numpy as np
from Rect import Rect
import os

FILEPATH = os.getcwd()
REJECTWIDTHFRAC =0.8
REJECTHEIGHTFRAC =0.8
DETECTREGION =190
AREAFACTOR =0.1
<<<<<<< HEAD

def showImage(name,img):
    imgr=cv2.resize(img,(600,800))
    cv2.imshow(name,imgr)

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
    print(r)
    while i<r:
        j=0
        while max(j,i)<r:
            if i!=j:
                #print(i,j,r,len(rects))
                if rects[i].isOverlapping(rects[j]) or rects[j].isOverlapping(rects[i]):
                    rects[i]=rects[i]+rects[j]
                    rects.remove(rects[j])
            j=j+1
            r=len(rects)
        i=i+1
        r=len(rects)
    return rects

# def doubleContourDetect(img,imgWidth,imgHeight):
#     for i in range(2):
#         img_g =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         r, img_t=cv2.threshold(img_g,DETECTREGION,255,0)
#         c,h=cv2.findContours(img_t,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#         if i==1: break
#         cv2.drawContours(img,c,-1,(255,0,0),3)

#     rects=contoursToBoundingRect(c,imgWidth,imgHeight)
#     rectsMerged = consecutiveRectMerge(rects)
#     rectsMerged = fullRectMerge(rects)

#     return rectsMerged
=======
TABLEUNIQUECOLOR=5
COLORREDUCER=64
>>>>>>> 155da628fe0d5ad7d521192209d5e8b9089e03bd

def doubleContourDetect(img_o):
    img=np.array(img_o)
    # imgc =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # imgc =imgc//COLORREDUCER*COLORREDUCER+COLORREDUCER//2
    
    imgWidth,imgHeight,imgC =img.shape
    rects=[]
    tables=[]
    images=[]
    def averageColor(pointSet):
        points =pointSet.flatten()
        colorSum =0
        n=len(points)
        for i in range(0,len(points),2):
            clr=img[points[i+1]][points[i]].astype('float')
            colorSum=colorSum+clr
        colorAverage= 2*colorSum//n
        return colorAverage.astype('uint8')

    def colorCount(rect:Rect)->int:
        p1=rect.lowerLeftPoint()
        p2=rect.upperRightPoint()
        imgd=imgc[p1[0]:p2[0],p1[1]:p2[1]]
        #imgd=np.reshape(imgd,(-1,3))
        u= np.unique(imgd,axis=0)
        return len(u)

    def showImage(name,img):
        imgr=cv2.resize(img,(600,800))
        cv2.imshow(name,imgr)

    

    for i in range(2):
        img_g =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_t=cv2.adaptiveThreshold(img_g,255,cv2.ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,11,12)
        #r, img_t=cv2.threshold(img_g,DETECTREGION,255,0)
        contours,hier=cv2.findContours(img_t,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #showImage(f'img{i}',img)
        #if i==1: break
        cv2.drawContours(img,contours,-1,(0,0,0),3)
        showImage(f'img{i}',img)

    imgc =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgc =imgc//COLORREDUCER*COLORREDUCER+COLORREDUCER//2

    showImage("imgc",imgc)

    for contour in contours:
        x,y,w,h= cv2.boundingRect(contour)
        newRect = Rect(x,y,w,h)
        if w<REJECTWIDTHFRAC*imgWidth and h<REJECTHEIGHTFRAC*imgHeight:
            contArea=cv2.contourArea(contour)
            rectUpdated=False
            if abs(contArea-w*h)<AREAFACTOR*w*h:
                c=colorCount(newRect)
                if c<=TABLEUNIQUECOLOR:
                    tables.append(newRect)
                else:
                    for rect in images:
                        if rect.isOverlapping(newRect):
                            rect.update(newRect)
                            rectUpdated=True
                            break
                    if not rectUpdated:
                        images.append(newRect)
                rectUpdated=True

            if not rectUpdated:
                for rect in images:
                    if rect.isOverlapping(newRect):
                        rect.update(newRect)
                        rectUpdated=True
                        break
                if not rectUpdated:
                    for rect in rects:
                        if rect.isOverlapping(newRect):
                            rect.update(newRect)
                            rectUpdated=True
                            break
                    if not rectUpdated:
                        rects.append(newRect)
    return {
        'rects':rects,
        'tables':tables,
        'images':images,
        'imgc':imgc,
    }
   

<<<<<<< HEAD
def detectByBlur(img_o):
    print("------------------------------Detecting by blurring-------------------------------")
    img=np.array(img_o)
    wid,hei,cha=img.shape
    img_b =cv2.GaussianBlur(img,(45,45),sigmaX=11,sigmaY=11)
    img_g =cv2.cvtColor(img_b,cv2.COLOR_BGR2GRAY)
    
    showImage("blurred",img_g)
    r, img_t1=cv2.threshold(img_g,225,255,0)
    showImage("thresh",img_t1)
    img_t=cv2.adaptiveThreshold(img_g,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,39,1)
    showImage("adaptiveThresh",img_t)
    c,h=cv2.findContours(img_t,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,c,-1,(255,0,0),3)
    showImage("contours only",img)
=======
# def detectByBlur(filename:str):
#     img = cv2.imread(FILEPATH+filename)
#     img_o=np.array(img)
#     wid,hei,cha=img.shape

#     img_g =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     img_b =cv2.GaussianBlur(img_g,(5,5),1  )
#     r, img_t=cv2.threshold(img_b,DETECTREGION,255,0)
#     c,h=cv2.findContours(img_t,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(img,c,-1,(255,0,0),3)
#     cv2.imshow("contours only",img)
>>>>>>> 155da628fe0d5ad7d521192209d5e8b9089e03bd

#     # img_g =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     # r, img_t=cv2.threshold(img_g,DETECTREGION,255,0)
#     # c,h=cv2.findContours(img_t,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

<<<<<<< HEAD
    # rects=contoursToBoundingRect(c,wid,hei)
    # rectsMerged = consecutiveRectMerge(rects)
    # rectsMerged = fullRectMerge(rects)
    # print(len(rectsMerged))

    # for rect in rectsMerged:
    #     cv2.rectangle(img_o,(rect.x,rect.y),(rect.x+rect.width,rect.y+rect.height),(0,0,255),2)

    # img_r = cv2.resize(img_o,(600,800))
    # cv2.imshow("img14o",img_r)
    print("-------------------------------------Detection Done------------------------------------")
=======
#     rects=contoursToBoundingRect(c,wid,hei)
#     rectsMerged = consecutiveRectMerge(rects)
#     rectsMerged = fullRectMerge(rects)
#     print(len(rectsMerged))

#     for rect in rectsMerged:
#         print(rect)
#         cv2.rectangle(img_o,(rect.x,rect.y),(rect.x+rect.width,rect.y+rect.height),(0,0,255),2)

#     img_r = cv2.resize(img_o,(600,800))
#     cv2.imshow("img14o",img_r)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
>>>>>>> 155da628fe0d5ad7d521192209d5e8b9089e03bd

def detect(img_o,detectFunc=doubleContourDetect):
    #img = cv2.imread(FILEPATH+FILEPREFIX+filename)
    img=np.array(img_o)

    # img_r = cv2.resize(img_o,(600,800))
    # cv2.imshow("img",img_r)
    def colorCount(rect:Rect)->int:
        p1=rect.lowerLeftPoint()
        p2=rect.upperRightPoint()
        imgd=imgc[p1[0]:p2[0],p1[1]:p2[1]]
        #imgd=np.reshape(imgd,(-1,3))
        u= np.unique(imgd,axis=0)
        return len(u)

    imgWidth,imgHeight,imgChannel=img.shape
<<<<<<< HEAD
    rects=detectFunc(img)

    # for rect in rects:
    #     cv2.rectangle(img_o,(rect.x,rect.y),(rect.x+rect.width,rect.y+rect.height),(0,0,255),2)
=======
    allrects=detectFunc(img)
    rects=allrects['rects']
    tables=allrects['tables']
    images=allrects['images']
    imgc=allrects['imgc']
    #print(rects)

    img_r = cv2.resize(img_o,(600,800))
    cv2.imshow("img_detcted",img_r)
    cv2.waitKey(0)

    for rect in rects:
        #print(rect)
        cv2.rectangle(img_o,(rect.lowerLeftPoint()[0],rect.lowerLeftPoint()[1]),(rect.upperRightPoint()[0],rect.upperRightPoint()[1]),(0,0,255),2)
        print(colorCount(rect))
        img_r = cv2.resize(img_o,(600,800))
        cv2.imshow("img_detcted",img_r)
        cv2.waitKey(0)

    for table in tables:
        cv2.rectangle(img_o,(table.lowerLeftPoint()[0],table.lowerLeftPoint()[1]),(table.upperRightPoint()[0],table.upperRightPoint()[1]),(0,255,0),2)
        print(colorCount(table))
        img_r = cv2.resize(img_o,(600,800))
        cv2.imshow("img_detcted",img_r)
        cv2.waitKey(0)

    for table in images:
        cv2.rectangle(img_o,(table.lowerLeftPoint()[0],table.lowerLeftPoint()[1]),(table.upperRightPoint()[0],table.upperRightPoint()[1]),(255,0,0),2)
        print(colorCount(table))
        img_r = cv2.resize(img_o,(600,800))
        cv2.imshow("img_detcted",img_r)
        cv2.waitKey(0) 
>>>>>>> 155da628fe0d5ad7d521192209d5e8b9089e03bd

    showImage("img_detcted",img_o)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {
        "height":imgHeight,
        "width":imgWidth,
        "rects":rects
    }

#doubleContourDetect("024.jpg")
#detectByBlur("024.jpg")
<<<<<<< HEAD
#detector("024.jpg")
img = cv2.imread(FILEPATH+"/test/out15.jpg")
detect(img,detectByBlur)
=======
#detector("024.jpg")  #images\Gr9_Science_and_Technology_NP_CDC_1st_2079BS-page-003.jpg
img = cv2.imread(FILEPATH+"/images/Gr9_Science_and_Technology_NP_CDC_1st_2079BS-page-023.jpg")
detect(img)
>>>>>>> 155da628fe0d5ad7d521192209d5e8b9089e03bd
