from Rect import Rect
import numpy as np
from DynamicNP import DynamicNpy
from num import Num
import math

class RectSet():
    def __init__(self,dimension:int|tuple,blockSize=200,blockMode=True,source=None) -> None:
        self.__blocksize = blockSize
        self.__blockMode = blockMode
        self.__source = source
        self.__image = DynamicNpy(Rect)
        if isinstance(dimension,int):
            dimension =(dimension,dimension)
        self.__dimension = dimension
        if blockMode:
            dimension = (math.ceil(dimension[0]/blockSize),math.ceil(dimension[1]/blockSize))
            self.__indexArray = np.array([None]*dimension[0]*dimension[1]).reshape(dimension).astype(DynamicNpy)
    def __eq__(self, o: object) -> bool:
        return self.__image.__eq__(o.__image)

    def addRect (self,rect:Rect):
        if self.__blockMode:
            if rect.__type != "Table":
                #TODO
                pass
            blocks = rect.getblocks(self.__blocksize)
            for block in blocks:
                if self.__indexArray[block]==None:
                    self.__indexArray[block]=DynamicNpy(Num)
                else:
                    for i, index in enumerate(self.__indexArray[block]):
                        if self.__image[index.num] and rect.overlapUpdate(self.__image[index.num]):
                            self.__image.removeIndex(index.num)
                            self.__indexArray[block].removeIndex(i)
            index = self.__image.append(rect)
            index = Num(index)
            for block in blocks:
                self.__indexArray[block].append(index)

        #else:
            #TODO
    def __iter__(self):
        yield from self.__image

    def __repr__(self) -> str:
        return self.__image.__repr__()
        pass
