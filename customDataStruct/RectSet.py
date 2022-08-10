from customDataStruct.Rect import Rect
import numpy as np
from customDataStruct.DynamicNP import DynamicNpy
from customDataStruct.num import Num #TODO remove this dependency
import math

class RectSet():
    """This class represents a set of rectangles, specifically designed to check and avoid 
        overlapping if necessary.
    """
    def __init__(self, dimension:int|tuple, blockSize=200, blockMode=True, source=None) -> None:
        """AI is creating summary for __init__

        Args:
            dimension (int|tuple): [description]
            blockSize (int, optional): [description]. Defaults to 200.
            blockMode (bool, optional): [description]. Defaults to True.
            source (str|np.ndarray, optional): [description]. Defaults to None.
        """
        self.__blocksize = blockSize
        self.__blockMode = blockMode
        self.__source = source
        self.__image = DynamicNpy(Rect)
        if isinstance(dimension,int):
            dimension =(dimension,dimension)
        self.__dimension = dimension
        if blockMode:
            dimension = (math.ceil(dimension[0]/blockSize)+2,math.ceil(dimension[1]/blockSize)+2)
            self.__indexArray = np.array([None]*dimension[0]*dimension[1]).reshape(dimension).astype(DynamicNpy)

    def __eq__(self, o: object) -> bool:
        """AI is creating summary for __eq__

        Args:
            o (object): [description]

        Returns:
            bool: [description]
        """
        return self.__image.__eq__(o.__image)

    def addRect (self,rect:Rect,padding=0):
        """AI is creating summary for addRect

        Args:
            rect (Rect): [description]
            padding (int, optional): [description]. Defaults to 0.
        """
        def updateRect(blocks):
            """AI is creating summary for updateRect

            Args:
                blocks ([type]): [description]

            Returns:
                [type]: [description]
            """
            for block in blocks:
                if self.__indexArray[block]==None:
                    self.__indexArray[block]=DynamicNpy(Num)
                else:
                    for i, index in enumerate(self.__indexArray[block]):
                        if self.__image[index()] and self.__image[index()].type!='Table' and rect.overlapUpdate(self.__image[index()],padding):
                            self.__image.removeIndex(index())
                            self.__indexArray[block].removeIndex(i)
                            blocks = rect.getblocks(self.__blocksize)
                            blocks = updateRect(blocks)
                            return blocks
            return blocks

        if self.__blockMode:
            blocks = rect.getblocks(self.__blocksize)
            if rect.type != "Table":
                #TODO
                blocks = updateRect(blocks)
                # for block in blocks:
                #     if self.__indexArray[block]==None:
                #         self.__indexArray[block]=DynamicNpy(Num)
                #     else:
                #         for i, index in enumerate(self.__indexArray[block]):
                #             if self.__image[index.num] and self.__image[index.num].type!='Table' and rect.overlapUpdate(self.__image[index.num],padding):
                #                 self.__image.removeIndex(index.num)
                #                 self.__indexArray[block].removeIndex(i)
            index = self.__image.append(rect)
            index = Num(index)
            for block in blocks:
                if self.__indexArray[block]==None:
                    self.__indexArray[block]=DynamicNpy(Num)
                self.__indexArray[block].append(index)

        #else:
            #TODO
    def __iter__(self):
        """AI is creating summary for __iter__

        Yields:
            [type]: [description]
        """
        yield from self.__image

    def __repr__(self) -> str:
        """AI is creating summary for __repr__

        Returns:
            str: [description]
        """
        return self.__image.__repr__()
