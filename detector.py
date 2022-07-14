import cv2
from cv2 import THRESH_BINARY
import numpy as np
from RectSet import RectSet
from Rect import Rect
from utility import *
import os

#global constants and factors
FILEPATH = os.getcwd()

def colorSpread(img8, pointSet:np.ndarray):
    #TODO use dynamicNPy instead of list
    points =pointSet.flatten()
    colors = []
    n=len(points)
    for i in range(0,n,2):
        clr=img8[points[i+1]][points[i]][0]
        colors.append(clr)
    return np.std(np.array(colors))

def doubleContourDetect(img_o,imgSource=None):
    img=np.array(img_o)
    img8=img//64*64+32 
    img8 = cv2.cvtColor(img8,cv2.COLOR_BGR2HSV)

    imgWidth,imgHeight,imgC =img.shape

    extractedData = RectSet((imgWidth,imgHeight),source=imgSource)
 
    #TODO use cascading and remove this loop
    for i in range(2):
        img_g =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img_t=cv2.adaptiveThreshold(img_g,255,cv2.ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,45,12) #TODO fix the parameters
        r, img_t=cv2.threshold(img_g,DETECTREGION,255,1)
        showImage(f"thresh{i}",img_t)
        contours,hier=cv2.findContours(img_t,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #if i==1: break
        cv2.drawContours(img,contours,-1,(255,0,0),i+3)#TODO contour color should be decided. A good approach is to draw inverse of image background
        showImage(f'img{i}',img)
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
            extractedData.addRect(newRect)

    print('---------------------------------------All contour processed--------------------------')

    return extractedData
   

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
    #detect(img_b)
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

def detect(img_o,detectFunc=doubleContourDetect):
    img=np.array(img_o)
    allrects=detectFunc(img)

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
    
    #print(allrects)
    rects=allrects
    # tables=allrects['tables']
    # images=allrects['images']
    #imgc=allrects['imgc']
    #showImage("imgc",imgc)
    #print(rects)

    showImage("img_detcted",img_o)
    cv2.waitKey(0)
    # for i, r in enumerate(rects):
    #     for j, r1 in enumerate(rects):
    #         if i!=j and r.isOverlapping(r1):
    #             print(r,r1)
    print("printed")
    img8=img//128*128+64 
    img8 = cv2.cvtColor(img8,cv2.COLOR_BGR2HSV)

    for rect in rects:
        #print(rect)
        colr = (0,0,255)
        if rect.type == 'Table': 
            colr = (0,255,0)
        elif rect.type =='Image':
            colr = (255,0,0)
        cv2.rectangle(img_o,(rect.lowerLeftPoint()[0],rect.lowerLeftPoint()[1]),(rect.upperRightPoint()[0],rect.upperRightPoint()[1]),colr,2)
        #print(colorCount(rect))
        showImage("img_detcted",img_o)
        cv2.waitKey(0)

    # for table in tables:
    #     cv2.rectangle(img_o,(table.lowerLeftPoint()[0],table.lowerLeftPoint()[1]),(table.upperRightPoint()[0],table.upperRightPoint()[1]),(0,255,0),2)
    #     #print(colorCount(table))
    #     showImage("img_detcted",img_o)
    #     cv2.waitKey(0)

    # for table in images:
    #     cv2.rectangle(img_o,(table.lowerLeftPoint()[0],table.lowerLeftPoint()[1]),(table.upperRightPoint()[0],table.upperRightPoint()[1]),(255,0,0),2)
    #     #print(colorCount(table))
    #     showImage("img_detcted",img_o)
    #     cv2.waitKey(0) 

    img_r = cv2.resize(img_o,(600,800))
    cv2.imshow("img_detcted",img_r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {
        "height":imgHeight,
        "width":imgWidth,
        "rects":rects
    }

#doubleContourDetect("024.jpg")
#detectByBlur("024.jpg")
#detector("024.jpg")  #images\Gr9_Science_and_Technology_NP_CDC_1st_2079BS-page-003.jpg

def sizeTransform(img):
    w,h,c=img.shape
    if w>IMGMAXSIZE:
        h=IMGMAXSIZE*h//w
        w=IMGMAXSIZE
    if h>IMGMAXSIZE:
        w=IMGMAXSIZE*w//h
        h=IMGMAXSIZE
    w=w//32*32
    h=h//32*32
    imgp = cv2.resize(img,(w,h))
    print(w,h)
    return imgp

gblur = lambda im: 2*cv2.GaussianBlur(im,(25,25),25,sigmaY=25)
blur = lambda im: 2*cv2.blur(im,(45,45))
colorReduce = lambda im: im//COLORREDUCER*COLORREDUCER+COLORREDUCER//2



def preprocess(img):
    
    imgp=cascading([blur,colorReduce],img)
    return imgp

def conv(img):
    dim=45
    kern = np.ones((dim,dim))/dim**2
    imgr =cv2.filter2D(img,ddepth=-1,kernel=kern)
    return imgr

def detect2(img):
    showImage('original',img)
    #showImage('preprocesed',preprocess(img))
    imgc = colorReduce(img)
    showImage("8color",imgc)
    showImage("convol",conv(imgc))
    detect(conv(imgc))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread("D:/OCR_Project/train/out23.jpg")
#img = cv2.imread(FILEPATH+"/images/Gr9_Science_and_Technology_NP_CDC_1st_2079BS-page-023.jpg")
detectByBlur(img)
