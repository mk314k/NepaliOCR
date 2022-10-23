"""
Dynamically resizing NumPy object array with free-list slot reuse.
"""
from typing import Iterator, List, Optional
import numpy as np


class DynamicNpy:
    """
    Typed dynamic array backed by a NumPy object buffer.

    Deleted indices are pushed onto `__nullIndex` and reused on subsequent `append`
    calls so external index references remain stable when other elements are removed.
    """

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
        """Insert `value`, reusing a freed slot if available, and return its index."""
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
        """Mark `index` as empty and push it onto the free-list for reuse."""
        if index < 0 or index >= self.size():
            raise IndexError(
                f"Index out of bounds: size is {self.size()} while {index} was requested"
            )
        if self.__array[index] is not None:
            self.__array[index] = None
            self.__nullIndex.append(index)

    def size(self) -> int:
        """High-water mark of allocated slots (including freed holes)."""
        return self.__lastValueIndex + 1

    def count(self) -> int:
        """Number of active (non-null) elements currently stored."""
        return sum(
            1
            for i in range(self.size())
            if isinstance(self.__array[i], self.__objType)
        )

    def __len__(self) -> int:
        return self.count()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DynamicNpy):
            return False
        return list(iter(self)) == list(iter(other))

    def __iter__(self) -> Iterator[object]:
        for i in range(self.size()):
            obj = self.__array[i]
            if isinstance(obj, self.__objType):
                yield obj

    def __repr__(self) -> str:
        return repr([obj for obj in self])
