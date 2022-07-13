from Rect import Rect
import numpy as np
from DynamicNP import DynamicNpy
from num import Num

class RectSet():
    def __init__(self,dimension:int|tuple,blockSize=200,blockMode=True) -> None:
        self.__blocksize = blockSize
        self.__blockMode = blockMode
        self.image = DynamicNpy(Rect)
        if isinstance(dimension,int):
            dimension =(dimension,dimension)
        self.__dimension = dimension
        if blockMode:
            dimension = (dimension[0]//blockSize+1,dimension[1]//blockSize+1)
            self.indexArray = np.array([None]*dimension[0]*dimension[1]).reshape(dimension).astype(DynamicNpy)
    def __eq__(self, o: object) -> bool:
        return self.image.__eq__(o.image)

    def addRect (self,rect:Rect):
        if self.__blockMode:
            blocks = rect.getblocks(self.__blocksize)
            for block in blocks:
                if self.indexArray[block]==None:
                    self.indexArray[block]=DynamicNpy(Num)
                else:
                    for i, index in enumerate(self.indexArray[block]):
                        if rect.overlapUpdate(self.image[index.num]):
                            self.image.removeIndex(index.num)
                            self.indexArray[block].removeIndex(i)
            index = self.image.append(rect)
            index = Num(index)
            for block in blocks:
                self.indexArray[block].append(index)

        #else:
            #TODO
    def __iter__(self):
        yield from self.image

    def __repr__(self) -> str:
        return self.image.__repr__()
        pass








