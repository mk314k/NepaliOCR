"""
Document layout segmentation using iterative contour thickening and projection profiles.
"""
from typing import Callable, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from nepali_ocr.spatial.rect import Rect
from nepali_ocr.spatial.rect_set import RectSet
from nepali_ocr.utils import (
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


def getPeaks(allLines: np.ndarray, maxLength: int) -> List[int]:
    """Return boundary indices where whitespace scanlines (`lineW >= maxLength`) transition into ink."""
    peakLines: List[int] = []
    prevI = 0
    for lineI, lineW in enumerate(allLines):
        if lineW >= maxLength:
            if lineI - prevI > 1:
                if prevI != -2:
                    peakLines.append(prevI)
                peakLines.append(lineI)
            prevI = lineI
    return peakLines


def getAllLines(
    imgo: np.ndarray, maxDim: Callable[[np.ndarray], int] = lambda img: img.shape[1]
) -> Tuple[List[int], np.ndarray]:
    """Compute horizontal projection profile and return line-boundary row pairs."""
    maxdim = maxDim(imgo)
    imgg = (
        cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
        if imgo.ndim == 3
        else imgo.copy()
    )
    _, imgt = cv2.threshold(imgg, DETECTREGION, 255, cv2.THRESH_BINARY_INV)
    imgt = 255 - imgt

    allLines = np.count_nonzero(imgt, axis=1)
    peakLines = getPeaks(allLines, maxdim)
    return peakLines, imgt


def detectLines(img: np.ndarray) -> RectSet:
    """Segment horizontal text-line bounding boxes via horizontal and vertical projection profiles."""
    imgHeight, imgWidth = img.shape[:2]
    peakLines, imgt = getAllLines(img)
    extractedRects = RectSet((imgHeight, imgWidth))

    for i in range(0, len(peakLines) - 1, 2):
        y0, y1 = peakLines[i], peakLines[i + 1]
        if y1 <= y0:
            continue
        imgtt = imgt[y0:y1, :]
        vlines = np.count_nonzero(imgtt, axis=0)
        vpeaks = getPeaks(vlines, y1 - y0)
        if len(vpeaks) >= 2:
            rect = Rect(vpeaks[0], y0, vpeaks[-1], y1)
            extractedRects.addRect(rect)

    return extractedRects


def detectBylines(img: np.ndarray) -> RectSet:
    """Segment word-level bounding boxes by slicing each horizontal line strip along vertical gaps."""
    imgHeight, imgWidth = img.shape[:2]
    peakLines, imgt = getAllLines(img)
    extractedRects = RectSet((imgHeight, imgWidth))

    for i in range(0, len(peakLines) - 1, 2):
        y0, y1 = peakLines[i], peakLines[i + 1]
        if y1 <= y0:
            continue
        imgtt = imgt[y0:y1, :]
        vlines = np.count_nonzero(imgtt, axis=0)
        vpeaks = getPeaks(vlines, y1 - y0)
        for j in range(0, len(vpeaks) - 1, 2):
            if vpeaks[j + 1] > vpeaks[j]:
                rect = Rect(vpeaks[j], y0, vpeaks[j + 1], y1)
                extractedRects.addRect(rect)

    return extractedRects


def detect(
    imgo: np.ndarray,
    padding: int = 0,
    iteration: int = 1,
    return_lines: bool = False,
):
    """Detect layout regions by running contour detection within projection-profile line strips."""
    img = np.array(imgo, copy=True)
    lines = detectLines(img)
    imgHeight, imgWidth = img.shape[:2]
    extractedRects = RectSet((imgHeight, imgWidth))

    if len(lines) > 0:
        for rect in lines:
            origin = rect["upperLeft"]
            end = rect["lowerRight"]
            imgtt = img[end[1] : origin[1], origin[0] : end[0]]
            if imgtt.size > 0:
                contourRectDetect(
                    imgtt,
                    iteration=iteration,
                    padding=padding,
                    rectsContainer=extractedRects,
                    translateVector=(origin[0], end[1]),
                )
    else:
        extractedRects = doubleContourDetect(
            img, padding=padding, iteration=max(2, iteration)
        )

    if return_lines:
        return extractedRects, lines
    return extractedRects


def annotateDetectedRegions(img: np.ndarray, rects: RectSet) -> np.ndarray:
    """Draw colour-coded bounding boxes: red for `'text'`, blue for `'Image'`, green for `'Table'`."""
    canvas = np.array(img, copy=True)
    for rect in rects:
        if rect.type == "Table":
            colr = (0, 255, 0)
        elif rect.type == "Image":
            colr = (255, 0, 0)
        else:
            colr = (0, 0, 255)
        cv2.rectangle(canvas, rect["lowerLeft"], rect["upperRight"], colr, 2)
    return canvas


if __name__ == "__main__":
    sample_path = "assets/gr9sctrain23_detected.jpg"
    img = cv2.imread(sample_path)
    if img is not None:
        rects = doubleContourDetect(img)
        annotated = annotateDetectedRegions(img, rects)
        showImage("image", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
