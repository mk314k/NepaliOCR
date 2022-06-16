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
    def __init__(self,x,y,width,height) -> None:
        self.lowerLeftVertex=Point(x,y)
        self.x=x
        self.y=y
        self.height=height
        self.width=width

    def __eq__(self, __o: object) -> bool:
        return (self.x==__o.x)and(self.y==__o.y)and(self.height==__o.height)and(self.width==__o.width)
        pass
    """
    @param rect another rectangle
    @returns if both rectangle
    """    
    def isOverlapping(self,rect:object)->bool:
        tf = lambda p:p[0]>=self.x and p[1]>=self.y and p[0]<=self.x+self.width and p[1]<=self.y+self.height
        return tf((rect.x,rect.y)) or tf((rect.x+rect.width,rect.y)) or tf((rect.x,rect.y+rect.height)) or tf((rect.x+rect.width,rect.y+rect.height))
    def __add__(self,rect:object)->object:
        x=min(self.x,rect.x)
        y=min(self.y,rect.y)
        return Rect(
            x,
            y,
            max(self.x+self.width,rect.x+rect.width)-x,
            max(self.y+self.height,rect.y+rect.height)-y
        )
    def __sub__(self,rect:object)->object:
        x=max(self.x,rect.x)
        y=max(self.y,rect.y)
        return Rect(
            x,
            y,
            min(self.x+self.width,rect.x+rect.width)-x,
            min(self.y+self.height,rect.y+rect.height)-y
        )
    def __str__(self):
        return [self.x,self.y,self.width,self.height].__str__()

