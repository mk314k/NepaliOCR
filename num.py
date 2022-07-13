class Num():
    def __init__(self,num:int) -> None:
        self.num=num
        pass
    def __str__(self) -> str:
        return self.num.__str__()
    def __repr__(self) -> str:
        return self.num.__repr__()
        pass