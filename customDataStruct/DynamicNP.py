from typing import Iterator, List, Optional
import numpy as np


class DynamicNpy:
    def __init__(
        self,
        objectType: type,
        initlen: int = 2,
        allowedObjType: Optional[type] = None,
    ) -> None:
        self.__array = np.empty(max(1, initlen), dtype=object)
        self.__array[:] = None
        self.__nullIndex: List[int] = []
        self.__objType = objectType
        self.__lastValueIndex = -1
        self.__allowedObjType = allowedObjType

    def __getitem__(self, index: int):
        if index < 0 or index >= self.size():
            raise IndexError(
                f"Index out of bounds: size is {self.size()} while {index} was requested"
            )
        return self.__array[index]

    def __setitem(self, index: int, value: object) -> int:
        if index >= self.__array.size:
            new_cap = max(2 * index, self.__array.size * 2, 2)
            arr = np.empty(new_cap, dtype=object)
            arr[:] = None
            arr[: self.__array.size] = self.__array
            self.__array = arr

        self.__array[index] = value
        if index > self.__lastValueIndex:
            self.__lastValueIndex = index
        return index

    def __setitem__(self, index: int, value: object) -> None:
        if value is not None and not isinstance(value, self.__objType):
            if self.__allowedObjType and isinstance(value, self.__allowedObjType):
                value = self.__objType(value)
            else:
                raise TypeError(
                    f"Expected {self.__objType.__name__}, got {type(value).__name__}"
                )
        self.__setitem(index, value)

    def append(self, value: object) -> int:
        if not isinstance(value, self.__objType):
            if self.__allowedObjType and isinstance(value, self.__allowedObjType):
                value = self.__objType(value)
            else:
                raise TypeError(
                    f"ValueType Error: expected {self.__objType.__name__}, got {type(value).__name__}"
                )
        if self.__nullIndex:
            index = self.__nullIndex.pop()
        else:
            index = self.__lastValueIndex + 1
        return self.__setitem(index, value)

    def removeIndex(self, index: int) -> None:
        if index < 0 or index >= self.size():
            raise IndexError(
                f"Index out of bounds: size is {self.size()} while {index} was requested"
            )
        if self.__array[index] is not None:
            self.__array[index] = None
            self.__nullIndex.append(index)

    def size(self) -> int:
        return self.__lastValueIndex + 1

    def __iter__(self) -> Iterator[object]:
        for i in range(self.size()):
            obj = self.__array[i]
            if isinstance(obj, self.__objType):
                yield obj

    def __repr__(self) -> str:
        return repr([obj for obj in self])
