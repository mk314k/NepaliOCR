from Point import Point
from interfaces import Rectangle
class Rect():
    """
    @param x is x coordinate of lower left vertex of the rectangle
    @param y is y coordinate of lower left vertex of the rectangle
    @param width is width of rectangle
    @param height is height of the rectangle
    @constructs an instance of rectangle 
    """
    def __init__(self,x1:float,y1:float,x2=0,y2=0,width=None,height=None,contArea=None,colr=None) -> None:
        self.__x1 = x1
        self.__y1 = y1
        secondPntInit = lambda pntWidth: pntWidth[1] if pntWidth[2]==None else pntWidth[0]+pntWidth[2]
        self.__x2:float = secondPntInit((x1,x2,width))
        self.__y2:float = secondPntInit((y1,y2,height))
        self.colr=colr
        self.contArea=contArea

    def width(self):
        return self.__x2 - self.__x1
    def height(self):
        return self.__y2 - self.__y1


    def area(self):
        return self.width()*self.height()


    def __eq__(self, o: object) -> bool:
        return (self.__x1==o.__x1)and(self.__x2==o.__x2)and(self.__y1==o.__y1)and (self.__y2 == o.__y2)

    def lowerLeftPoint(self)->Point:
        return (self.__x1,self.__y1)

    # def lowerRightPoint(self)->Point:
    #     return self.__origin+Point(self.__width,0)

    # def upperLeftPoint(self)->Point:
    #     return self.__origin+Point(0,self.__height)

    def upperRightPoint(self)->Point:
        return (self.__x2,self.__y2)

    def __getitem__(self,index):
        itemList = [self.lowerLeftPoint,self.lowerRightPoint,self.upperRightPoint,self.upperLeftPoint]
        return itemList[index]()
    """
    @param rect another rectangle
    @returns if both rectangle
    """    
    def isOverlapping(self,rect,padding=0):
        if isinstance(padding,int):
            padding=(padding,padding)
        return not(
            self.__x1 > rect.__x2 + 2*padding[0] or 
            self.__y1 > rect.__y2 + 2*padding[1] or 
            self.__x2 < rect.__x1 + 2*padding[0] or 
            self.__y2 < rect.__y1 + 2*padding[1])

    # def __contains__(self,point:Point)->bool:
    #     return point.isUpperRightTo(self.__origin) and self.upperRightPoint().isUpperRightTo(point)

    def update(self,rect:Rectangle)->None:
        self.__x1 = min(self.__x1, rect.__x1)
        self.__y1 = min(self.__y1, rect.__y1)
        self.__x2 = max(self.__x2, rect.__x2)
        self.__y2 = max(self.__y2, rect.__y2)

    def overlapUpdate(self,rect:Rectangle,padding=0)->bool:
        result = self.isOverlapping(rect,padding)
        if result: self.update(rect)
        return result

    def getblocks(self,blocksize):
        x1 = self.__x1//blocksize
        y1 = self.__y1//blocksize
        x2 = self.__x2//blocksize+1
        y2 = self.__y2//blocksize+1
        blocks=[]
        for x in range(x1,x2):
            for y in range(y1,y2):
                blocks.append((x,y))
        return blocks


    # def __add__(self,rect:Rectangle)->Rectangle:
    #     minOrigin=self.__origin.minXYCompare(rect.__origin)
    #     dimension = self.upperRightPoint().maxXYCompare(rect[2])
    #     return Rect(
    #         minOrigin[0],
    #         minOrigin[1],
    #         dimension[0]-minOrigin[0],
    #         dimension[1]-minOrigin[1]
    #     )
    # def __sub__(self,rect:Rectangle)->Rectangle:
    #     __x=max(self.__x,rect.__x)
    #     y=max(self.y,rect.y)
    #     return Rect(
    #         __x,
    #         y,
    #         min(self.x+self.width,rect.x+rect.width)-x,
    #         min(self.y+self.height,rect.y+rect.height)-y
    #     )
    def __str__(self)->str:
        return self.__repr__()
    def __repr__(self) -> str:
        return f"({self.__x1},{self.__y1})[]({self.__x2},{self.__y2})"
