from __future__ import annotations
import math
from typing import Iterator, List, Optional, Tuple, Union
import numpy as np
from customDataStruct.DynamicNP import DynamicNpy
from customDataStruct.num import Num
from customDataStruct.Rect import Rect


class RectSet:
    def __init__(
        self,
        dimension: Union[int, Tuple[int, int]],
        blockSize: int = 200,
        blockMode: bool = True,
        source: Optional[Union[str, np.ndarray]] = None,
    ) -> None:
        self.__blocksize = blockSize
        self.__blockMode = blockMode
        self.__source = source
        self.__image = DynamicNpy(Rect)
        if isinstance(dimension, int):
            dimension = (dimension, dimension)
        self.__dimension = (int(dimension[0]), int(dimension[1]))
        if blockMode:
            grid_dim = (
                math.ceil(self.__dimension[0] / blockSize) + 2,
                math.ceil(self.__dimension[1] / blockSize) + 2,
            )
            self.__indexArray = np.empty(grid_dim, dtype=object)
            self.__indexArray[:] = None

    def _clamp_blocks(
        self, blocks: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        max_r, max_c = self.__indexArray.shape
        return [
            (r, c) for r, c in blocks if 0 <= r < max_r and 0 <= c < max_c
        ]

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, RectSet):
            return False
        return self.__image == o.__image

    def addRect(
        self, rect: Rect, padding: Union[int, Tuple[int, int]] = 0
    ) -> None:
        def updateRect(
            blocks: List[Tuple[int, int]]
        ) -> List[Tuple[int, int]]:
            for block in blocks:
                cell = self.__indexArray[block]
                if cell is None:
                    self.__indexArray[block] = DynamicNpy(Num)
                else:
                    for i in range(cell.size()):
                        index = cell[i]
                        if index is None:
                            continue
                        idx_val = index()
                        if idx_val < 0 or idx_val >= self.__image.size():
                            continue
                        existing = self.__image[idx_val]
                        if (
                            existing is not None
                            and existing.type != "Table"
                            and rect.overlapUpdate(existing, padding)
                        ):
                            old_blocks = self._clamp_blocks(
                                existing.getblocks(self.__blocksize)
                            )
                            self.__image.removeIndex(idx_val)
                            for ob in old_blocks:
                                ob_cell = self.__indexArray[ob]
                                if ob_cell is not None:
                                    for j in range(ob_cell.size()):
                                        entry = ob_cell[j]
                                        if entry is not None and entry() == idx_val:
                                            ob_cell.removeIndex(j)
                            new_blocks = self._clamp_blocks(
                                rect.getblocks(self.__blocksize)
                            )
                            return updateRect(new_blocks)
            return blocks

        if self.__blockMode:
            blocks = self._clamp_blocks(rect.getblocks(self.__blocksize))
            if rect.type != "Table":
                blocks = updateRect(blocks)
            index = Num(self.__image.append(rect))
            for block in blocks:
                if self.__indexArray[block] is None:
                    self.__indexArray[block] = DynamicNpy(Num)
                self.__indexArray[block].append(index)

    def __iter__(self) -> Iterator[Rect]:
        yield from self.__image

    def __repr__(self) -> str:
        return repr(self.__image)
