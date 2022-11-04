"""
Document layout segmentation using iterative contour thickening and projection profiles.
"""
from typing import Callable, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from customDataStruct.Rect import Rect
from customDataStruct.RectSet import RectSet
from utility import (
    AREAFACTOR,
    DETECTREGION,
    L1DistanceFrac,
    REJECTHEIGHTFRAC,
    REJECTWIDTHFRAC,
    colorCount,
    colorSpread,
    showImage,
)


def contourCorrection(
    imgo: np.ndarray, iteration: int = 2, adaptive: bool = False
) -> Tuple[np.ndarray, list]:
    """
    Threshold `imgo` and draw progressively thicker contours for `iteration` passes.

    Drawing contours back onto the working image bridges small horizontal gaps along
    the Devanagari *shirorekha* (top bar) and attaches upper/lower vowel matras to
    their parent word box.
    """
    img = np.array(imgo, copy=True)
    contours = []
    for i in range(max(1, iteration)):
        imgg = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.ndim == 3
            else img.copy()
        )
        if adaptive:
            imgt = cv2.adaptiveThreshold(
                imgg,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                15,
                8,
            )
        else:
            _, imgt = cv2.threshold(
                imgg, DETECTREGION, 255, cv2.THRESH_BINARY_INV
            )
        contours, _ = cv2.findContours(
            imgt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, contours, -1, (255, 0, 0), i + 2)
    return img, list(contours)


def mergeAndClassify(
    contours: Sequence,
    img8: np.ndarray,
    padding: int = 0,
    imgSource: Optional[str] = None,
    rectsContainer: Optional[RectSet] = None,
    translateVector: Tuple[int, int] = (0, 0),
) -> RectSet:
    """
    Convert OpenCV contours into `Rect` objects, classify each as `'text'`, `'Image'`,
    or `'Table'` based on rectangular fill ratio and quantized HSV color spread, and
    insert into a `RectSet`.
    """
    imgHeight, imgWidth = img8.shape[:2]
    extractedData = (
        rectsContainer
        if rectsContainer is not None
        else RectSet((imgHeight, imgWidth), source=imgSource)
    )
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 2 or h <= 2:
            continue
        newRect = Rect(
            x + translateVector[0],
            y + translateVector[1],
            width=w,
            height=h,
        )
        if w < REJECTWIDTHFRAC * imgWidth and h < REJECTHEIGHTFRAC * imgHeight:
            cspr = colorSpread(img8, contour)
            c_area = cv2.contourArea(contour)
            if L1DistanceFrac((newRect.area(), c_area)) < AREAFACTOR:
                newRect.type = "Table" if cspr == 0 else "Image"
            if newRect.type == "text" and cspr > 14:
                cCnt, _ = colorCount(newRect, img8)
                if cCnt > 12:
                    newRect.type = "Image"
            extractedData.addRect(newRect, padding)
    return extractedData


def doubleContourDetect(
    img_o: np.ndarray,
    padding: int = 0,
    iteration: int = 2,
    imgSource: Optional[str] = None,
) -> RectSet:
    """Run two-pass contour thickening followed by spatial merging and region classification."""
    img = np.array(img_o, copy=True)
    img8 = (img // 64) * 64 + 32
    img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV)
    img, contours = contourCorrection(img, iteration=iteration)
    return mergeAndClassify(contours, img8, padding, imgSource)


def contourRectDetect(
    img_o: np.ndarray,
    padding: int = 0,
    iteration: int = 2,
    imgSource: Optional[str] = None,
    rectsContainer: Optional[RectSet] = None,
    translateVector: Tuple[int, int] = (0, 0),
) -> Optional[RectSet]:
    """Detect filled bounding rectangles within a sub-region and merge into `rectsContainer`."""
    img = np.array(img_o, copy=True)
    img8 = (img // 64) * 64 + 32
    img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV)

    img, contours = contourCorrection(img, iteration=iteration)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), -1)
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_t = cv2.threshold(img_g, DETECTREGION, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        img_t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if rectsContainer is not None:
        mergeAndClassify(
            contours,
            img8,
            padding,
            imgSource,
            rectsContainer=rectsContainer,
            translateVector=translateVector,
        )
        return rectsContainer

    return mergeAndClassify(contours, img8, padding, imgSource)


def customDetector(rectDetection: Callable):
    return lambda img: rectDetection(img)


