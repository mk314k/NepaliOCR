from interfaces import Rectangle
import math
import cv2
import numpy as np


class Rect():
    """AI is creating summary for Rect
    """
    def __init__(self,x1:float,y1:float,x2=0,y2=0,width=None,height=None,type='text',colr=None) -> None:
        """AI is creating summary for __init__

        Args:
            x1 (float): [description]
            y1 (float): [description]
            x2 (int, optional): [description]. Defaults to 0.
            y2 (int, optional): [description]. Defaults to 0.
            width ([type], optional): [description]. Defaults to None.
            height ([type], optional): [description]. Defaults to None.
            type (str, optional): [description]. Defaults to 'text'.
            colr ([type], optional): [description]. Defaults to None.
        """
        self.__x1 = x1
        self.__y1 = y1
        secondPntInit = lambda pointData: pointData[1] if pointData[2]==None else pointData[0]+pointData[2]
        self.__x2:float = secondPntInit((x1,x2,width))
        self.__y2:float = secondPntInit((y1,y2,height))
        self.colr=colr
        self.type=type
        #TODO maybe replace type with mergeAllowed:bool

    def width(self):
        return self.__x2 - self.__x1

    def height(self):
        return self.__y2 - self.__y1

    def area(self):
        return self.width()*self.height()


    def __eq__(self, o: object) -> bool:
        return (self.__x1==o.__x1)and(self.__x2==o.__x2)and(self.__y1==o.__y1)and (self.__y2 == o.__y2)

    def __getitem__(self,index):
        itemList = {'lowerLeft':(self.__x1,self.__y1), 
                    'lowerRight':(self.__x2,self.__y1),
                    'upperRight':(self.__x2,self.__y2),
                    'upperLeft':(self.__x1,self.__y2),
                    'width':self.width(),
                    'height':self.height(),
                    'area':self.area()}
        return itemList[index]
   
    def isOverlapping(self,rect,padding=0):
        if isinstance(padding,int):
            padding=(padding,padding)
        return not(
            self.__x1 > rect.__x2 + 2*padding[0] or 
            self.__y1 > rect.__y2 + 2*padding[1] or 
            self.__x2 < rect.__x1 - 2*padding[0] or 
            self.__y2 < rect.__y1 - 2*padding[1])

    def update(self,rect:Rectangle,padding=0, paddedUpdate = True)->None:
        if not paddedUpdate:
            padding=0
        self.__x1 = min(self.__x1, rect.__x1)-padding
        self.__y1 = min(self.__y1, rect.__y1)-padding
        self.__x2 = max(self.__x2, rect.__x2)+padding
        self.__y2 = max(self.__y2, rect.__y2)+padding
        if rect.type == 'Image':
            self.type = 'Image'

    def overlapUpdate(self,rect:Rectangle,padding=0)->bool:
        result = self.isOverlapping(rect,padding)
        if result: self.update(rect)
        return result

    def getblocks(self,blocksize):
        x1 = self.__x1//blocksize
        y1 = self.__y1//blocksize
        x2 = math.ceil(self.__x2/blocksize)
        y2 = math.ceil(self.__y2/blocksize)
        return [(y,x) for y in range(y1,y2) for x in range(x1,x2)]

    def __str__(self)->str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"({self.__x1},{self.__y1})[]({self.__x2},{self.__y2})"


    

