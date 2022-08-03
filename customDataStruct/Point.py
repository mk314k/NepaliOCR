class Point():
    def __init__(self,x:float,y:float):
        self.__x=x
        self.__y=y

    def __getitem__(self,index:int)->float:
        if index==0:
            return self.__x
        elif index==1:
            return self.__y

    def __str__(self)->str:
        return (self.__x,self.__y).__str__()

    def __add__(self,point:object)->object:
        return Point(self.__x+point[0],self.__y+point[1])

    def __radd__(self,point:object)->object:
        return Point(self.__x+point[0],self.__y+point[1])

    def isUpperRightTo(self,point:object)->bool:
        return self.__x>=point.__x and self.__y>=point.__y

    def minXYCompare(self,point):
        return Point(
            min(self.__x,point.__x),
            min(self.__y,point.__y)
        )
    def maxXYCompare(self,point):
        return Point(
            max(self.__x,point.__x),
            max(self.__y,point.__y)
        )
