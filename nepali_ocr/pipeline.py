"""
End-to-end page and PDF OCR pipeline combining `detector.py` and `recognizer.py`.
"""
import os
from typing import Dict, List, Optional
import cv2
import numpy as np

from nepali_ocr.detector import detect, double_contour_detect
from nepali_ocr.io_pdf import PDFReader, PDFWriter
from nepali_ocr.recognizer import recognize


def process_image_page(
    img: np.ndarray, model_path: str = "", modelPath: Optional[str] = None
) -> Dict[str, List[dict]]:
    """
    Detect layout regions on `img` and run OCR on all `'text'` boxes in reading order.
    Returns a dictionary grouped by region type (`'text'`, `'Image'`, `'Table'`).
    """
    if modelPath is not None:
        model_path = modelPath
    extracted_data = double_contour_detect(img)
    summary: Dict[str, List[dict]] = {"text": [], "Image": [], "Table": []}

    for data_rect in extracted_data.reading_order():
        if data_rect.type == "text":
            if data_rect.width() < 8 or data_rect.height() < 10:
                continue
            crop = data_rect.crop_image(img, pad=1)
            txt = (
                recognize(crop, model_path=model_path)
                if model_path
                else recognize(crop)
            )
            summary["text"].append({"rect": data_rect, "text": txt})
        elif data_rect.type == "Image":
            summary["Image"].append({"rect": data_rect, "crop": data_rect.crop_image(img)})
        elif data_rect.type == "Table":
            summary["Table"].append({"rect": data_rect, "crop": data_rect.crop_image(img)})
    return summary


processImagePage = process_image_page


def process_pdf(file_name: str) -> PDFWriter:
    """Rasterize `file_name`, run layout detection + OCR per page, and write `<name>-out.pdf`."""
    pdf_in = PDFReader(file_name, mode="r")
    img_names = pdf_in.to_image()
    pdf_out = PDFWriter(file_name[:-4] + "-out.pdf", mode="w")

    for img_name in img_names:
        pdf_out.new_page()
        img = cv2.imread(img_name)
        if img is None:
            continue
        extracted_data = detect(img)

        for data_rect in extracted_data.reading_order():
            if data_rect.type == "text":
                recognized_text = recognize(data_rect.crop_image(img, pad=1))
                pdf_out.add_text(data_rect, recognized_text)
            elif data_rect.type == "Image":
                pdf_out.add_image(data_rect, img)
            elif data_rect.type == "Table":
                pdf_out.add_table(data_rect, img)

    pdf_out.close()
    return pdf_out


main = process_pdf


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        target = sys.argv[1]
        if target.lower().endswith(".pdf"):
            process_pdf(target)
        else:
            page_img = cv2.imread(target)
            res = process_image_page(page_img)
            print(
                f"Detected {len(res['text'])} text regions, "
                f"{len(res['Image'])} images, {len(res['Table'])} tables."
            )
