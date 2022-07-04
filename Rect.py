from Point import Point
"""
A rectangular frame
"""
class Rect():
    """
    @param x is x coordinate of lower left vertex of the rectangle
    @param y is y coordinate of lower left vertex of the rectangle
    @param width is width of rectangle
    @param height is height of the rectangle
    @constructs an instance of rectangle 
    """
    def __init__(self,x:float,y:float,width:float,height:float) -> None:
        self.__origin=Point(x,y)
        self.__height=height
        self.__width=width

    def __eq__(self, __o: object) -> bool:
        return (self.__origin==__o.__origin)and(self.__height==__o.__height)and(self.__width==__o.__width)

    def lowerLeftPoint(self)->Point:
        return self.__origin

    def lowerRightPoint(self)->Point:
        return self.__origin+Point(self.__width,0)

    def upperLeftPoint(self)->Point:
        return self.__origin+Point(0,self.__height)

    def upperRightPoint(self)->Point:
        return self.__origin+Point(self.__width,self.__height)


    def __getitem__(self,index):
        if index==0:
            return self.lowerLeftPoint()

    """
    @param rect another rectangle
    @returns if both rectangle
    """    
    def isOverlapping(self,rect:object)->bool:
        verticesContained = self.lowerLeftPoint() in rect or self.lowerRightPoint() in rect or self.upperLeftPoint() in rect or self.upperRightPoint() in rect
        containsVertices = self.__contains__(rect.lowerLeftPoint()) or self.__contains__(rect.lowerRightPoint()) or self.__contains__(rect.upperLeftPoint()) or self.__contains__(rect.upperRightPoint())
        return verticesContained or containsVertices

    def __contains__(self,point:Point)->bool:
        return point.isUpperRightTo(self.__origin) and self.upperRightPoint().isUpperRightTo(point)

    def update(self,rect:object)->None:
        rect2 = self.__add__(rect)
        self.__origin=rect2.__origin
        self.__height=rect2.__height
        self.__width=rect2.__width

    def __add__(self,rect:object)->object:
        minOrigin=self.__origin.minXYCompare(rect.__origin)
        dimension = self.upperRightPoint().maxXYCompare(rect.upperRightPoint())
        return Rect(
            minOrigin[0],
            minOrigin[1],
            dimension[0]-minOrigin[0],
            dimension[1]-minOrigin[1]
        )
    # def __sub__(self,rect:object)->object:
    #     __x=max(self.__x,rect.__x)
    #     y=max(self.y,rect.y)
    #     return Rect(
    #         __x,
    #         y,
    #         min(self.x+self.width,rect.x+rect.width)-x,
    #         min(self.y+self.height,rect.y+rect.height)-y
    #     )
    def __str__(self):
        return "origin: "+self.__origin.__str__()+f" width:{self.__width} and height: {self.__height} "

