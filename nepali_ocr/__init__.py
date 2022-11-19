from nepali_ocr.detector import (
    annotate_detected_regions,
    annotateDetectedRegions,
    detect,
    detect_lines,
    detect_words_by_line,
    detectBylines,
    detectLines,
    double_contour_detect,
    doubleContourDetect,
)
from nepali_ocr.io_pdf import PDFReader, PDFWriter, open_pdf, superPDF
from nepali_ocr.model import CRNN
from nepali_ocr.pipeline import process_image_page, process_pdf, processImagePage
from nepali_ocr.recognizer import (
    ctc_greedy_decode,
    postProcess,
    preProcess,
    preprocess_crop,
    recognize,
)
from nepali_ocr.spatial import (
    DynamicArray,
    DynamicNpy,
    Num,
    Point,
    Rect,
    Rectangle,
    RectSet,
)
from nepali_ocr.utils import (
    decode_indices,
    encode_text,
    preeti2Uni,
    preeti_to_unicode,
    unicodes,
)

__all__ = [
    "Rectangle",
    "Num",
    "Point",
    "DynamicArray",
    "DynamicNpy",
    "Rect",
    "RectSet",
    "PDFReader",
    "PDFWriter",
    "open_pdf",
    "superPDF",
    "detect",
    "double_contour_detect",
    "doubleContourDetect",
    "detect_lines",
    "detectLines",
    "detect_words_by_line",
    "detectBylines",
    "annotate_detected_regions",
    "annotateDetectedRegions",
    "CRNN",
    "preprocess_crop",
    "preProcess",
    "ctc_greedy_decode",
    "postProcess",
    "recognize",
    "process_image_page",
    "processImagePage",
    "process_pdf",
    "preeti_to_unicode",
    "preeti2Uni",
    "unicodes",
    "encode_text",
    "decode_indices",
]
