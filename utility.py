import os
from typing import Dict, List, Tuple
import cv2
import numpy as np
from customDataStruct.Rect import Rect

# Rectangle corner indices
lowerLeft = 0
upperLeft = 1
upperRight = 2
lowerRight = 3

# Layout detection thresholds
REJECTWIDTHFRAC = 0.8
REJECTHEIGHTFRAC = 0.8
DETECTREGION = 190
AREAFACTOR = 0.1
TABLEUNIQUECOLOR = 5
COLORREDUCER = 96
IMGMAXSIZE = 1024

L1Distance = lambda p1p2: abs(p1p2[1] - p1p2[0])
L1DistanceFrac = lambda p1p2: L1Distance(p1p2) / max(1e-9, abs(p1p2[0]))


def showImage(name: str, img: np.ndarray) -> None:
    if os.environ.get("NEPALI_OCR_HEADLESS", "1") == "1":
        return
    imgr = cv2.resize(img, (750, 1000))
    cv2.imshow(name, imgr)


def colorSpread(img8: np.ndarray, pointSet: np.ndarray) -> float:
    points = pointSet.flatten()
    if len(points) < 2:
        return 0.0
    h, w = img8.shape[:2]
    colors = []
    for i in range(0, len(points), 2):
        px = min(max(0, int(points[i])), w - 1)
        py = min(max(0, int(points[i + 1])), h - 1)
        clr = img8[py, px, 0] if img8.ndim == 3 else img8[py, px]
        colors.append(float(clr))
    return float(np.std(np.array(colors)))


def colorCount(rect: Rect, imgc: np.ndarray) -> Tuple[int, float]:
    crop = rect.getImage(imgc)
    if crop.size == 0:
        return 0, 0.0
    flat = crop.reshape(-1, crop.shape[-1]) if crop.ndim == 3 else crop.flatten()
    u = np.unique(flat, axis=0)
    return len(u), float(np.std(flat))


def cascading(funcs, img: np.ndarray, repeat: int = 1) -> np.ndarray:
    imgp = np.array(img)
    while repeat > 0:
        for func in funcs:
            imgp = func(imgp)
        repeat -= 1
    return imgp


def boundingObject(contour: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    pts = contour.reshape(-1, 2)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    return (int(min_xy[0]), int(min_xy[1])), (int(max_xy[0]), int(max_xy[1]))
