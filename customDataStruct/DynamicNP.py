import numpy as np

class DynamicNpy():
    
    def __init__(self,objectType:type,initlen=2,allowedObjType=None) -> None:
        """AI is creating summary for __init__

        Args:
            objectType ([type]): [description]
            initlen (int, optional): [description]. Defaults to 2.
        """
        self.__array = np.array([None]*initlen).astype(object)
        self.__nullIndex = []
        self.__objType = objectType
        self.__lastValueIndex =-1
        self.__allowedObjType = allowedObjType

    def __getitem__(self,index:int):
        """AI is creating summary for __getitem__

        Args:
            index ([type]): [description]

        Returns:
            [type]: [description]
        """
        if (index<0 or index>=self.size()):
            raise Exception(f"Index out of bound. Array is size of {self.size()} while {index} is called")
        return self.__array[index]

    def __setitem(self,index:int,value:object):
        # if not isinstance(value,self.__objType):
        #     if isinstance(value,self.__allowedObjType):
        #         value=self.__objType(value)
        #     else:
        #         raise Exception("This array is not defined to store given value object type")

        if index >= self.__array.size:
            arr = np.array([None]*2*index).astype(self.__objType)
            arr[0:self.__array.size] = self.__array
            self.__array = arr

        self.__array[index]=value
        if index>self.__lastValueIndex:
            self.__lastValueIndex=index
        return index

    # def __setitem__(self,index,value):
    #     _=self.__setitem(index,value)

    def append(self,value):
        if not isinstance(value,self.__objType):
            raise Exception("ValueType Error")
        index = self.__lastValueIndex + 1
        if self.__nullIndex:
            index = self.__nullIndex.pop()
        return self.__setitem(index,value)

    def removeIndex(self,index):
        if (index<0 or index>=self.size()):
            raise Exception(f"Index out of bound. Array is size of {self.size()} while {index} is called")
        self.__array[index]=None
        self.__nullIndex.append(index)
        #TODO
        # if len(self.__nullIndex)>=self.size()//2:
        #     arr = 

    def size(self):
        return self.__lastValueIndex+1

    def __iter__(self):
        for obj in self.__array:
            if isinstance(obj,self.__objType):
                yield obj

    def __repr__(self) -> str:
        return self.__array.__repr__()
        pass
