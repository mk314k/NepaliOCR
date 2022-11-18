"""
Dynamically resizing NumPy object array with free-list slot reuse.
"""
from typing import Iterator, List, Optional
import numpy as np


class DynamicArray:
    """
    Typed dynamic array backed by a NumPy object buffer.

    Deleted indices are pushed onto `__free_indices` and reused on subsequent `append`
    calls so external index references remain stable when other elements are removed.
    """

    def __init__(
        self,
        object_type: type,
        init_len: int = 2,
        allowed_obj_type: Optional[type] = None,
        initlen: Optional[int] = None,
        allowedObjType: Optional[type] = None,
    ) -> None:
        if initlen is not None:
            init_len = initlen
        if allowedObjType is not None:
            allowed_obj_type = allowedObjType
        self.__array = np.empty(max(1, init_len), dtype=object)
        self.__array[:] = None
        self.__free_indices: List[int] = []
        self.__obj_type = object_type
        self.__last_value_index = -1
        self.__allowed_obj_type = allowed_obj_type

    def __getitem__(self, index: int):
        if index < 0 or index >= self.size():
            raise IndexError(
                f"Index out of bounds: size is {self.size()} while {index} was requested"
            )
        return self.__array[index]

    def _set_slot(self, index: int, value: object) -> int:
        if index >= self.__array.size:
            new_cap = max(2 * index, self.__array.size * 2, 2)
            arr = np.empty(new_cap, dtype=object)
            arr[:] = None
            arr[: self.__array.size] = self.__array
            self.__array = arr

        self.__array[index] = value
        if index > self.__last_value_index:
            self.__last_value_index = index
        return index

    def __setitem__(self, index: int, value: object) -> None:
        if value is not None and not isinstance(value, self.__obj_type):
            if self.__allowed_obj_type and isinstance(
                value, self.__allowed_obj_type
            ):
                value = self.__obj_type(value)
            else:
                raise TypeError(
                    f"Expected {self.__obj_type.__name__}, got {type(value).__name__}"
                )
        self._set_slot(index, value)

    def append(self, value: object) -> int:
        """Insert `value`, reusing a freed slot if available, and return its index."""
        if not isinstance(value, self.__obj_type):
            if self.__allowed_obj_type and isinstance(
                value, self.__allowed_obj_type
            ):
                value = self.__obj_type(value)
            else:
                raise TypeError(
                    f"ValueType Error: expected {self.__obj_type.__name__}, got {type(value).__name__}"
                )
        if self.__free_indices:
            index = self.__free_indices.pop()
        else:
            index = self.__last_value_index + 1
        return self._set_slot(index, value)

    def remove_index(self, index: int) -> None:
        """Mark `index` as empty and push it onto the free-list for reuse."""
        if index < 0 or index >= self.size():
            raise IndexError(
                f"Index out of bounds: size is {self.size()} while {index} was requested"
            )
        if self.__array[index] is not None:
            self.__array[index] = None
            self.__free_indices.append(index)

    removeIndex = remove_index

    def size(self) -> int:
        """High-water mark of allocated slots (including freed holes)."""
        return self.__last_value_index + 1

    def count(self) -> int:
        """Number of active (non-null) elements currently stored."""
        return sum(
            1
            for i in range(self.size())
            if isinstance(self.__array[i], self.__obj_type)
        )

    def __len__(self) -> int:
        return self.count()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DynamicArray):
            return False
        return list(iter(self)) == list(iter(other))

    def __iter__(self) -> Iterator[object]:
        for i in range(self.size()):
            obj = self.__array[i]
            if isinstance(obj, self.__obj_type):
                yield obj

    def __repr__(self) -> str:
        return repr([obj for obj in self])


DynamicNpy = DynamicArray
