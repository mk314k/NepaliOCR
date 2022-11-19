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
    REJECTHEIGHTFRAC,
    REJECTWIDTHFRAC,
    color_count,
    color_spread,
    l1_distance_frac,
    show_image,
)


def contour_correction(
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


contourCorrection = contour_correction


def merge_and_classify(
    contours: Sequence,
    img8: np.ndarray,
    padding: int = 0,
    img_source: Optional[str] = None,
    rects_container: Optional[RectSet] = None,
    translate_vector: Tuple[int, int] = (0, 0),
    imgSource: Optional[str] = None,
    rectsContainer: Optional[RectSet] = None,
    translateVector: Optional[Tuple[int, int]] = None,
) -> RectSet:
    """
    Convert OpenCV contours into `Rect` objects, classify each as `'text'`, `'Image'`,
    or `'Table'` based on rectangular fill ratio and quantized HSV color spread, and
    insert into a `RectSet`.
    """
    if imgSource is not None:
        img_source = imgSource
    if rectsContainer is not None:
        rects_container = rectsContainer
    if translateVector is not None:
        translate_vector = translateVector

    img_height, img_width = img8.shape[:2]
    extracted_data = (
        rects_container
        if rects_container is not None
        else RectSet((img_height, img_width), source=img_source)
    )
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 2 or h <= 2:
            continue
        new_rect = Rect(
            x + translate_vector[0],
            y + translate_vector[1],
            width=w,
            height=h,
        )
        if w < REJECTWIDTHFRAC * img_width and h < REJECTHEIGHTFRAC * img_height:
            cspr = color_spread(img8, contour)
            c_area = cv2.contourArea(contour)
            if l1_distance_frac((new_rect.area(), c_area)) < AREAFACTOR:
                new_rect.type = "Table" if cspr == 0 else "Image"
            if new_rect.type == "text" and cspr > 14:
                c_cnt, _ = color_count(new_rect, img8)
                if c_cnt > 12:
                    new_rect.type = "Image"
            extracted_data.add_rect(new_rect, padding)
    return extracted_data


mergeAndClassify = merge_and_classify


def double_contour_detect(
    img_o: np.ndarray,
    padding: int = 0,
    iteration: int = 2,
    img_source: Optional[str] = None,
    imgSource: Optional[str] = None,
) -> RectSet:
    """Run two-pass contour thickening followed by spatial merging and region classification."""
    if imgSource is not None:
        img_source = imgSource
    img = np.array(img_o, copy=True)
    img8 = (img // 64) * 64 + 32
    img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV)
    img, contours = contour_correction(img, iteration=iteration)
    return merge_and_classify(contours, img8, padding, img_source)


doubleContourDetect = double_contour_detect


def contour_rect_detect(
    img_o: np.ndarray,
    padding: int = 0,
    iteration: int = 2,
    img_source: Optional[str] = None,
    rects_container: Optional[RectSet] = None,
    translate_vector: Tuple[int, int] = (0, 0),
    imgSource: Optional[str] = None,
    rectsContainer: Optional[RectSet] = None,
    translateVector: Optional[Tuple[int, int]] = None,
) -> Optional[RectSet]:
    """Detect filled bounding rectangles within a sub-region and merge into `rects_container`."""
    if imgSource is not None:
        img_source = imgSource
    if rectsContainer is not None:
        rects_container = rectsContainer
    if translateVector is not None:
        translate_vector = translateVector

    img = np.array(img_o, copy=True)
    img8 = (img // 64) * 64 + 32
    img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV)

    img, contours = contour_correction(img, iteration=iteration)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), -1)
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_t = cv2.threshold(img_g, DETECTREGION, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        img_t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if rects_container is not None:
        merge_and_classify(
            contours,
            img8,
            padding,
            img_source,
            rects_container=rects_container,
            translate_vector=translate_vector,
        )
        return rects_container

    return merge_and_classify(contours, img8, padding, img_source)


contourRectDetect = contour_rect_detect


def custom_detector(rect_detection: Callable):
    return lambda img: rect_detection(img)


customDetector = custom_detector


def get_peaks(all_lines: np.ndarray, max_length: int) -> List[int]:
    """Return boundary indices where whitespace scanlines (`line_w >= max_length`) transition into ink."""
    peak_lines: List[int] = []
    prev_i = 0
    for line_i, line_w in enumerate(all_lines):
        if line_w >= max_length:
            if line_i - prev_i > 1:
                if prev_i != -2:
                    peak_lines.append(prev_i)
                peak_lines.append(line_i)
            prev_i = line_i
    return peak_lines


getPeaks = get_peaks


def get_all_lines(
    imgo: np.ndarray, max_dim: Callable[[np.ndarray], int] = lambda img: img.shape[1]
) -> Tuple[List[int], np.ndarray]:
    """Compute horizontal projection profile and return line-boundary row pairs."""
    maxdim = max_dim(imgo)
    imgg = (
        cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
        if imgo.ndim == 3
        else imgo.copy()
    )
    _, imgt = cv2.threshold(imgg, DETECTREGION, 255, cv2.THRESH_BINARY_INV)
    imgt = 255 - imgt

    all_lines = np.count_nonzero(imgt, axis=1)
    peak_lines = get_peaks(all_lines, maxdim)
    return peak_lines, imgt


getAllLines = get_all_lines


def detect_lines(img: np.ndarray) -> RectSet:
    """Segment horizontal text-line bounding boxes via horizontal and vertical projection profiles."""
    img_height, img_width = img.shape[:2]
    peak_lines, imgt = get_all_lines(img)
    extracted_rects = RectSet((img_height, img_width))

    for i in range(0, len(peak_lines) - 1, 2):
        y0, y1 = peak_lines[i], peak_lines[i + 1]
        if y1 <= y0:
            continue
        imgtt = imgt[y0:y1, :]
        vlines = np.count_nonzero(imgtt, axis=0)
        vpeaks = get_peaks(vlines, y1 - y0)
        if len(vpeaks) >= 2:
            rect = Rect(vpeaks[0], y0, vpeaks[-1], y1)
            extracted_rects.add_rect(rect)

    return extracted_rects


detectLines = detect_lines


def detect_words_by_line(img: np.ndarray) -> RectSet:
    """Segment word-level bounding boxes by slicing each horizontal line strip along vertical gaps."""
    img_height, img_width = img.shape[:2]
    peak_lines, imgt = get_all_lines(img)
    extracted_rects = RectSet((img_height, img_width))

    for i in range(0, len(peak_lines) - 1, 2):
        y0, y1 = peak_lines[i], peak_lines[i + 1]
        if y1 <= y0:
            continue
        imgtt = imgt[y0:y1, :]
        vlines = np.count_nonzero(imgtt, axis=0)
        vpeaks = get_peaks(vlines, y1 - y0)
        for j in range(0, len(vpeaks) - 1, 2):
            if vpeaks[j + 1] > vpeaks[j]:
                rect = Rect(vpeaks[j], y0, vpeaks[j + 1], y1)
                extracted_rects.add_rect(rect)

    return extracted_rects


detectBylines = detect_words_by_line


def detect(
    imgo: np.ndarray,
    padding: int = 0,
    iteration: int = 1,
    return_lines: bool = False,
):
    """Detect layout regions by running contour detection within projection-profile line strips."""
    img = np.array(imgo, copy=True)
    lines = detect_lines(img)
    img_height, img_width = img.shape[:2]
    extracted_rects = RectSet((img_height, img_width))

    if len(lines) > 0:
        for rect in lines:
            origin = rect["upperLeft"]
            end = rect["lowerRight"]
            imgtt = img[end[1] : origin[1], origin[0] : end[0]]
            if imgtt.size > 0:
                contour_rect_detect(
                    imgtt,
                    iteration=iteration,
                    padding=padding,
                    rects_container=extracted_rects,
                    translate_vector=(origin[0], end[1]),
                )
    else:
        extracted_rects = double_contour_detect(
            img, padding=padding, iteration=max(2, iteration)
        )

    if return_lines:
        return extracted_rects, lines
    return extracted_rects


def annotate_detected_regions(img: np.ndarray, rects: RectSet) -> np.ndarray:
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


annotateDetectedRegions = annotate_detected_regions

if __name__ == "__main__":
    sample_path = "assets/gr9sctrain23_detected.jpg"
    img = cv2.imread(sample_path)
    if img is not None:
        rects = double_contour_detect(img)
        annotated = annotate_detected_regions(img, rects)
        show_image("image", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
