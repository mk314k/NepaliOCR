import numpy as np

class DynamicNpy():
    def __init__(self,objectType,initlen=2) -> None:
        self.array = np.array([None]*initlen).astype(objectType)
        self.nullindex = []
        self.objType = objectType
        self.lastValueIndex =0
    def __getitem__(self,index):
        return self.array[index]
    def __setitem(self,index,value):
        if index >= self.array.size:
            arr = np.array([None]*2*index).astype(self.objType)
            arr[0:self.array.size] = self.array
            self.array = arr
        self.array[index]=value
        if index>=self.lastValueIndex:
            self.lastValueIndex=index+1
        return index
    def __setitem__(self,index,value):
        _=self.__setitem(index,value)
    def append(self,value):
        index = self.lastValueIndex
        if self.nullindex:
            index = self.nullindex.pop()
        return self.__setitem(index,value)
        pass
    def removeIndex(self,index):
        self.array[index]=None
        self.nullindex.append(index)
        pass
    def size(self):
        return self.array.size
    def __iter__(self):
        for obj in self.array:
            if isinstance(obj,self.objType):
                yield obj

    def __repr__(self) -> str:
        return self.array.__repr__()
        pass
