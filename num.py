class Num():
    def __init__(self,num:int) -> None:
        self.__num=num
    def __call__(self) -> int:
        return self.__num
    def __str__(self) -> str:
        return self.num.__str__()
    def __repr__(self) -> str:
        return self.num.__repr__()
