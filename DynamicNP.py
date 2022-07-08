import numpy as np

class DynamicNpy():
    def __init__(self,objectType,initlen=2) -> None:
        self.__array = np.array([None]*initlen).astype(objectType)
        self.__nullIndex = []
        self.__objType = objectType
        self.lastValueIndex =0
    def __getitem__(self,index):
        return self.__array[index]
    def __setitem__(self,index,value):
        if index >= self.__array.size:
            arr = np.array([None]*2*index).astype(self.__objType)
            arr[0:self.__array.size] = self.__array
            self.__array = arr
        self.__array[index]=value
        if index>=self.lastValueIndex:
            self.lastValueIndex=index+1
    def append(self,value):
        index = self.lastValueIndex
        if self.__nullIndex:
            index = self.__nullIndex.pop()
        self.__setitem__(index,value)
        pass
    def removeIndex(self,index):
        self.__array[index]=None
        self.__nullIndex.append(index)
        pass
    def size(self):
        return self.__array.size
    def __iter__(self):
        for obj in self.__array:
            if isinstance(obj,self.__objType):
                yield obj

    def __repr__(self) -> str:
        return self.__array.__repr__()
        pass
