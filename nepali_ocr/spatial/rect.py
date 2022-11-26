"""
Axis-aligned bounding box (`Rect`) for document layout regions.
"""
from __future__ import annotations
import math
from typing import List, Optional, Tuple, Union
import numpy as np
from nepali_ocr.spatial.interfaces import Rectangle


class Rect(Rectangle):
    """
    Bounding box defined by `(x1, y1)` (top-left in image coordinates) and
    `(x2, y2)` (bottom-right), tagged with a layout region `type`
    (`'text'`, `'Image'`, or `'Table'`).
    """

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

    @property
    def x1(self) -> int:
        return self.__x1

    @property
    def y1(self) -> int:
        return self.__y1

    @property
    def x2(self) -> int:
        return self.__x2

    @property
    def y2(self) -> int:
        return self.__y2

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
        if index in ("lowerLeft", "top_left", 0):
            return (self.__x1, self.__y1)
        if index in ("lowerRight", "top_right", 3):
            return (self.__x2, self.__y1)
        if index in ("upperRight", "bottom_right", 2):
            return (self.__x2, self.__y2)
        if index in ("upperLeft", "bottom_left", 1):
            return (self.__x1, self.__y2)
        if index == "width":
            return self.width()
        if index == "height":
            return self.height()
        if index == "area":
            return self.area()
        raise KeyError(f"Unknown Rect property: {index}")

    def is_overlapping(
        self, rect: Rect, padding: Union[int, float, Tuple[float, float]] = 0
    ) -> bool:
        """Check whether `self` and `rect` intersect when expanded by `padding`."""
        if isinstance(padding, (int, float)):
            pad_x, pad_y = float(padding), float(padding)
        else:
            pad_x, pad_y = float(padding[0]), float(padding[1])
        return not (
            self.__x1 > rect.__x2 + 2.0 * pad_x
            or self.__y1 > rect.__y2 + 2.0 * pad_y
            or self.__x2 < rect.__x1 - 2.0 * pad_x
            or self.__y2 < rect.__y1 - 2.0 * pad_y
        )

    isOverlapping = is_overlapping

    def intersection_area(self, rect: Rect) -> int:
        """Compute overlapping area in pixels between `self` and `rect`."""
        ix1 = max(self.__x1, rect.__x1)
        iy1 = max(self.__y1, rect.__y1)
        ix2 = min(self.__x2, rect.__x2)
        iy2 = min(self.__y2, rect.__y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0
        return (ix2 - ix1) * (iy2 - iy1)

    intersectionArea = intersection_area

    def iou(self, rect: Rect) -> float:
        """Intersection-over-Union (IoU) in `[0.0, 1.0]`."""
        inter = self.intersection_area(rect)
        if inter == 0:
            return 0.0
        union = self.area() + rect.area() - inter
        return inter / union if union > 0 else 0.0

    def update(
        self, rect: Rect, padding: int = 0, padded_update: bool = True
    ) -> None:
        """Expand `self` in-place to enclose `rect` (propagating `'Image'` type if present)."""
        if not padded_update:
            padding = 0
        self.__x1 = min(self.__x1, rect.__x1) - padding
        self.__y1 = min(self.__y1, rect.__y1) - padding
        self.__x2 = max(self.__x2, rect.__x2) + padding
        self.__y2 = max(self.__y2, rect.__y2) + padding
        if rect.type == "Image" or (
            self.type == "text" and self.width() >= 75 and self.height() >= 75
        ):
            self.type = "Image"

    def overlap_update(
        self, rect: Rect, padding: Union[int, Tuple[int, int]] = 0
    ) -> bool:
        """If `rect` overlaps `self`, merge `rect` into `self` and return True."""
        result = self.is_overlapping(rect, padding)
        if result:
            self.update(rect)
        return result

    overlapUpdate = overlap_update

    def crop_image(self, img: np.ndarray, pad: int = 0) -> np.ndarray:
        """Crop the image subarray `[y1:y2, x1:x2]` expanded by `pad` pixels and clamped to `img` bounds."""
        h, w = img.shape[:2]
        p = max(0, int(pad))
        x1 = max(0, min(w, self.__x1 - p))
        y1 = max(0, min(h, self.__y1 - p))
        x2 = max(x1, min(w, self.__x2 + p))
        y2 = max(y1, min(h, self.__y2 + p))
        return img[y1:y2, x1:x2]

    getImage = crop_image

    def get_blocks(self, block_size: int) -> List[Tuple[int, int]]:
        """Return `(row_block, col_block)` spatial grid cells covered by this rectangle."""
        x1 = max(0, self.__x1 // block_size)
        y1 = max(0, self.__y1 // block_size)
        x2 = max(x1 + 1, math.ceil(self.__x2 / block_size))
        y2 = max(y1 + 1, math.ceil(self.__y2 / block_size))
        return [(y, x) for y in range(y1, y2) for x in range(x1, x2)]

    getblocks = get_blocks

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"({self.__x1},{self.__y1})[]({self.__x2},{self.__y2})"
