"""
2D point primitive with componentwise comparison helpers.
"""
from __future__ import annotations
from typing import Union


class Point:
    """2D Cartesian coordinate `(x, y)`."""

    def __init__(self, x: float, y: float) -> None:
        self.__x = float(x)
        self.__y = float(y)

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.__x
        if index == 1:
            return self.__y
        raise IndexError(f"Point index {index} out of range (expected 0 or 1)")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (Point, tuple, list)) and len(other) >= 2:
            return self.__x == float(other[0]) and self.__y == float(other[1])
        return False

    def __len__(self) -> int:
        return 2

    def __str__(self) -> str:
        return str((self.__x, self.__y))

    def __repr__(self) -> str:
        return f"Point({self.__x}, {self.__y})"

    def __add__(self, point: Union[Point, tuple, list]) -> Point:
        return Point(self.__x + point[0], self.__y + point[1])

    def __radd__(self, point: Union[Point, tuple, list]) -> Point:
        return Point(self.__x + point[0], self.__y + point[1])

    def __sub__(self, point: Union[Point, tuple, list]) -> Point:
        return Point(self.__x - point[0], self.__y - point[1])

    def isUpperRightTo(self, point: Point) -> bool:
        """Return True if `self` lies at or above/right of `point`."""
        return self.__x >= point.__x and self.__y >= point.__y

    def minXYCompare(self, point: Point) -> Point:
        """Return componentwise minimum `(min(x1, x2), min(y1, y2))`."""
        return Point(min(self.__x, point.__x), min(self.__y, point.__y))

    def maxXYCompare(self, point: Point) -> Point:
        """Return componentwise maximum `(max(x1, x2), max(y1, y2))`."""
        return Point(max(self.__x, point.__x), max(self.__y, point.__y))
