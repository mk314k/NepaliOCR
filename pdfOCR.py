"""
End-to-end page and PDF OCR pipeline combining `detector.py` and `recognizer.py`.
"""
import os
from typing import Dict, List
import cv2
import numpy as np

from customDataStruct.pdfClass import PDFReader, PDFWriter
from detector import detect, doubleContourDetect
from recognizer import recognize


def processImagePage(
    img: np.ndarray, modelPath: str = ""
) -> Dict[str, List[dict]]:
    """
    Detect layout regions on `img` and run OCR on all `'text'` boxes in reading order.
    Returns a dictionary grouped by region type (`'text'`, `'Image'`, `'Table'`).
    """
    extractedData = doubleContourDetect(img)
    summary: Dict[str, List[dict]] = {"text": [], "Image": [], "Table": []}

    for dataRect in extractedData.readingOrder():
        crop = dataRect.getImage(img)
        if dataRect.type == "text":
            if dataRect.width() < 14 or dataRect.height() < 10:
                continue
            txt = (
                recognize(crop, modelPath=modelPath)
                if modelPath
                else recognize(crop)
            )
            summary["text"].append({"rect": dataRect, "text": txt})
        elif dataRect.type == "Image":
            summary["Image"].append({"rect": dataRect, "crop": crop})
        elif dataRect.type == "Table":
            summary["Table"].append({"rect": dataRect, "crop": crop})
    return summary


def main(fileName: str) -> PDFWriter:
    """Rasterize `fileName`, run layout detection + OCR per page, and write `<name>-out.pdf`."""
    pdfIn = PDFReader(fileName, mode="r")
    imgNames = pdfIn.toImage()
    pdfOut = PDFWriter(fileName[:-4] + "-out.pdf", mode="w")

    for imgName in imgNames:
        pdfOut.newPage()
        img = cv2.imread(imgName)
        if img is None:
            continue
        extractedData = detect(img)

        for dataRect in extractedData.readingOrder():
            if dataRect.type == "text":
                recognizedText = recognize(dataRect.getImage(img))
                pdfOut.addText(dataRect, recognizedText)
            elif dataRect.type == "Image":
                pdfOut.addImage(dataRect, img)
            elif dataRect.type == "Table":
                pdfOut.addTable(dataRect, img)

    pdfOut.close()
    return pdfOut


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        target = sys.argv[1]
        if target.lower().endswith(".pdf"):
            main(target)
        else:
            page_img = cv2.imread(target)
            res = processImagePage(page_img)
            print(
                f"Detected {len(res['text'])} text regions, "
                f"{len(res['Image'])} images, {len(res['Table'])} tables."
            )
