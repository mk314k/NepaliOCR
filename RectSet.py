from Rect import Rect
import numpy as np
from DynamicNP import DynamicNpy

class RectSet():
    def __init__(self,dimension:int|tuple,blockSize=200,blockMode=True) -> None:
        self.__blocksize = blockSize
        self.__blockMode = blockMode
        self.__image = DynamicNpy(Rect)
        if isinstance(dimension,int):
            dimension =(dimension,dimension)
        self.__dimension = dimension
        if blockMode:
            dimension = (dimension[0]//blockSize+1,dimension[1]//blockSize+1)
            self.__blocks = np.array([None]*dimension[0]*dimension[1]).reshape(dimension).astype(DynamicNpy)
        
    def __eq__(self, o: object) -> bool:
        return self.__image.__eq__(o.__image)

    def addRect (self,rect:Rect):
        indice = range(0,self.__image.size())
        if self.__blockMode:
            blocks = rect.getblocks()
            for block in blocks:
                indice = indice + self.__blocks[block] #TODO
        
        for index in indice:
            if self.__image[index]==None:
                self.__image[index]=DynamicNpy()
            else:
                for i, existingRect in enumerate(self.__image[index]):
                    if isinstance(existingRect,Rect) and existingRect.isOverlapping(rect):
                        rect.update(existingRect)
                        self.__image[block].removeIndex(i)
                        print("overlapping",rect,existingRect,self.__image)
            self.__image[block].append(rect)
        print(rect,self.__image)
    def __iter__(self):
        yield from self.__image

    def __repr__(self) -> str:
        return self.__image.__repr__()
        pass








