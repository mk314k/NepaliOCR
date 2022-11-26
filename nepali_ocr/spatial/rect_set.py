"""
Spatial-grid container (`RectSet`) for incrementally merging overlapping bounding boxes.
"""
from __future__ import annotations
import math
from typing import Iterator, List, Optional, Tuple, Union
import numpy as np
from nepali_ocr.spatial.dynamic_array import DynamicArray
from nepali_ocr.spatial.num import Num
from nepali_ocr.spatial.rect import Rect


class RectSet:
    """
    Collection of `Rect` regions partitioned into `block_size x block_size` grid cells.

    When a non-table `Rect` is added in `block_mode`, only candidate rectangles sharing
    at least one spatial block are checked for overlap. If an overlap occurs, the two
    boxes are merged into their union and grid checks repeat until no further overlaps
    exist.
    """

    def __init__(
        self,
        dimension: Union[int, Tuple[int, int]],
        block_size: int = 200,
        block_mode: bool = True,
        source: Optional[Union[str, np.ndarray]] = None,
        blockSize: Optional[int] = None,
        blockMode: Optional[bool] = None,
    ) -> None:
        if blockSize is not None:
            block_size = blockSize
        if blockMode is not None:
            block_mode = blockMode
        self.__blocksize = block_size
        self.__blockMode = block_mode
        self.__source = source
        self.__image = DynamicArray(Rect)
        if isinstance(dimension, int):
            dimension = (dimension, dimension)
        self.__dimension = (int(dimension[0]), int(dimension[1]))
        if self.__blockMode:
            grid_dim = (
                math.ceil(self.__dimension[0] / self.__blocksize) + 2,
                math.ceil(self.__dimension[1] / self.__blocksize) + 2,
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

    def add_rect(
        self, rect: Rect, padding: Union[int, Tuple[int, int]] = 0
    ) -> None:
        """Insert `rect`, merging transitive overlaps with existing non-table regions."""
        def update_rect(
            blocks: List[Tuple[int, int]]
        ) -> List[Tuple[int, int]]:
            for block in blocks:
                cell = self.__indexArray[block]
                if cell is None:
                    self.__indexArray[block] = DynamicArray(Num)
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
                            and rect.overlap_update(existing, padding)
                        ):
                            old_blocks = self._clamp_blocks(
                                existing.get_blocks(self.__blocksize)
                            )
                            self.__image.remove_index(idx_val)
                            for ob in old_blocks:
                                ob_cell = self.__indexArray[ob]
                                if ob_cell is not None:
                                    for j in range(ob_cell.size()):
                                        entry = ob_cell[j]
                                        if entry is not None and entry() == idx_val:
                                            ob_cell.remove_index(j)
                            new_blocks = self._clamp_blocks(
                                rect.get_blocks(self.__blocksize)
                            )
                            return update_rect(new_blocks)
            return blocks

        if self.__blockMode:
            blocks = self._clamp_blocks(rect.get_blocks(self.__blocksize))
            if rect.type != "Table":
                blocks = update_rect(blocks)
            index = Num(self.__image.append(rect))
            for block in blocks:
                if self.__indexArray[block] is None:
                    self.__indexArray[block] = DynamicArray(Num)
                self.__indexArray[block].append(index)
        else:
            if rect.type != "Table":
                merged = True
                while merged:
                    merged = False
                    for idx_val in range(self.__image.size()):
                        existing = self.__image[idx_val]
                        if (
                            existing is not None
                            and existing.type != "Table"
                            and rect.overlap_update(existing, padding)
                        ):
                            self.__image.remove_index(idx_val)
                            merged = True
                            break
            self.__image.append(rect)

    addRect = add_rect

    def reading_order(self, line_tolerance: int = 14) -> List[Rect]:
        """
        Group regions into horizontal text lines by vertical-midpoint proximity
        (`line_tolerance` px), sort lines top-to-bottom, and sort regions within
        each line strictly left-to-right (`x1`).
        """
        rects = list(iter(self))
        if not rects:
            return []

        # Sort by vertical center first to form horizontal line clusters
        by_cy = sorted(
            rects, key=lambda r: 0.5 * (r["lowerLeft"][1] + r["upperRight"][1])
        )
        lines: List[List[Rect]] = []
        line_centers: List[float] = []

        for r in by_cy:
            cy = 0.5 * (r["lowerLeft"][1] + r["upperRight"][1])
            placed = False
            for idx, mean_cy in enumerate(line_centers):
                if abs(cy - mean_cy) <= line_tolerance:
                    lines[idx].append(r)
                    n = len(lines[idx])
                    line_centers[idx] = mean_cy + (cy - mean_cy) / n
                    placed = True
                    break
            if not placed:
                lines.append([r])
                line_centers.append(cy)

        ordered: List[Rect] = []
        for idx in sorted(range(len(lines)), key=lambda i: line_centers[i]):
            line_rects = sorted(lines[idx], key=lambda r: r["lowerLeft"][0])
            ordered.extend(line_rects)
        return ordered

    readingOrder = reading_order

    def __len__(self) -> int:
        return len(self.__image)

    def __iter__(self) -> Iterator[Rect]:
        yield from self.__image

    def __repr__(self) -> str:
        return repr(self.__image)
