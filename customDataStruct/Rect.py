from __future__ import annotations
import math
from typing import List, Optional, Tuple, Union
from customDataStruct.interfaces import Rectangle


class Rect(Rectangle):
    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float = 0,
        y2: float = 0,
        width: Optional[float] = None,
        height: Optional[float] = None,
        type: str = "text",
        colr: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        raw_x2 = x2 if width is None else x1 + width
        raw_y2 = y2 if height is None else y1 + height
        self.__x1: int = int(min(x1, raw_x2))
        self.__y1: int = int(min(y1, raw_y2))
        self.__x2: int = int(max(x1, raw_x2))
        self.__y2: int = int(max(y1, raw_y2))
        self.colr = colr
        self.type = type

    def width(self) -> int:
        return self.__x2 - self.__x1

    def height(self) -> int:
        return self.__y2 - self.__y1

    def area(self) -> int:
        return max(0, self.width()) * max(0, self.height())

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Rect):
            return False
        return (
            self.__x1 == o.__x1
            and self.__x2 == o.__x2
            and self.__y1 == o.__y1
            and self.__y2 == o.__y2
        )

    def __getitem__(self, index: Union[str, int]):
        if index == "lowerLeft" or index == 0:
            return (self.__x1, self.__y1)
        if index == "lowerRight" or index == 3:
            return (self.__x2, self.__y1)
        if index == "upperRight" or index == 2:
            return (self.__x2, self.__y2)
        if index == "upperLeft" or index == 1:
            return (self.__x1, self.__y2)
        if index == "width":
            return self.width()
        if index == "height":
            return self.height()
        if index == "area":
            return self.area()
        raise KeyError(f"Unknown Rect property: {index}")

    def isOverlapping(
        self, rect: Rect, padding: Union[int, Tuple[int, int]] = 0
    ) -> bool:
        if isinstance(padding, (int, float)):
            pad_x, pad_y = int(padding), int(padding)
        else:
            pad_x, pad_y = int(padding[0]), int(padding[1])
        return not (
            self.__x1 > rect.__x2 + 2 * pad_x
            or self.__y1 > rect.__y2 + 2 * pad_y
            or self.__x2 < rect.__x1 - 2 * pad_x
            or self.__y2 < rect.__y1 - 2 * pad_y
        )

    def update(
        self, rect: Rect, padding: int = 0, paddedUpdate: bool = True
    ) -> None:
        if not paddedUpdate:
            padding = 0
        self.__x1 = min(self.__x1, rect.__x1) - padding
        self.__y1 = min(self.__y1, rect.__y1) - padding
        self.__x2 = max(self.__x2, rect.__x2) + padding
        self.__y2 = max(self.__y2, rect.__y2) + padding
        if rect.type == "Image":
            self.type = "Image"

    def overlapUpdate(
        self, rect: Rect, padding: Union[int, Tuple[int, int]] = 0
    ) -> bool:
        result = self.isOverlapping(rect, padding)
        if result:
            self.update(rect)
        return result

    def getblocks(self, blocksize: int) -> List[Tuple[int, int]]:
        x1 = max(0, self.__x1 // blocksize)
        y1 = max(0, self.__y1 // blocksize)
        x2 = max(x1 + 1, math.ceil(self.__x2 / blocksize))
        y2 = max(y1 + 1, math.ceil(self.__y2 / blocksize))
        return [(y, x) for y in range(y1, y2) for x in range(x1, x2)]

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"({self.__x1},{self.__y1})[]({self.__x2},{self.__y2})"
