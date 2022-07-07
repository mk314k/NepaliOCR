from Rect import Rect
import numpy as np

BLOCK_SIZE =50

class DynamicNpy():
    def __init__(self,objectType=Rect) -> None:
        self.array = np.zeros(10).astype(objectType)
        self.nullIndex = []
        self.lastValueIndex =0
    def __getitem__(self,index):
        return self.array[index]
    def __setitem__(self,index,value):
        if index >= self.array.size:
            arr = np.zeros(2*index)
            arr[0:self.array.size] = self.array
            self.array = arr
        self.array[index]=value
        if index>=self.lastValueIndex:
            self.lastValueIndex=index+1
    def append(self,value):
        index = self.lastValueIndex
        if self.nullIndex:
            index = self.nullIndex.pop()
        self.__setitem__(index,value)
        pass
    def remove(self,index):
        self.array[index]=0
        self.nullIndex.append(index)
        pass

class RectSet():
    def __init__(self,dimension:int|tuple) -> None:
        if isinstance(dimension,int):
            dimension =(dimension,dimension)
        dimension = (dimension[0]//BLOCK_SIZE+1,dimension[1]//BLOCK_SIZE+1)
        self.__image = np.zeros(dimension).astype(DynamicNpy)
        
    def __eq__(self, __o: object) -> bool:
        if self.__image.shape != __o.__image.shape : return False
        return True

    def addRect (self,rect:Rect):
        blocks = rect.getblocks(BLOCK_SIZE)
        for block in blocks:
            if self.__image[block]==0:
                self.__image[block]=set([rect])
            else:
                for i, existingRect in enumerate(self.__image[block]):
                    if existingRect.isOverlapping(rect):
                        rect =rect + existingRect
                        self.__image[block].remove(i)
            self.__image[block].append(rect)








