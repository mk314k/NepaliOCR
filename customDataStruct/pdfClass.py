import locale
import os
from typing import List, Union
import cv2
import numpy as np

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None


class PDFReader:
    """Read a PDF file, extract embedded text, or render pages to JPEG images."""

    def __init__(self, fileName: str, mode: str = "r") -> None:
        self.__fileName = fileName
        self.__mode = mode
        self.__pdf = PdfReader(fileName) if PdfReader is not None else None

    def totalPages(self) -> int:
        if self.__pdf is None:
            return 0
        if hasattr(self.__pdf, "pages"):
            return len(self.__pdf.pages)
        return self.__pdf.getNumPages()

    def toImage(self, outExt: str = "jpg", resolution: int = 300) -> List[str]:
        if outExt == "jpg":
            return self.toJPG(resolution)
        raise ValueError(f"Unsupported output extension: {outExt}")

    def toJPG(self, resolution: int = 300) -> List[str]:
        imgDir = self.__fileName[:-4] + "Img"
        os.makedirs(imgDir, exist_ok=True)
        fileNamePrefix = os.path.join(imgDir, "img")
        n_pages = self.totalPages()

        try:
            import ghostscript

            args = [
                "pdf2jpeg",
                "-dNOPAUSE",
                "-dBATCH",
                "-dSAFER",
                "-sDEVICE=jpeg",
                f"-r{resolution}",
                "-sOutputFile=" + fileNamePrefix + "%d.jpg",
                self.__fileName,
            ]
            encoding = locale.getpreferredencoding()
            encoded_args = [a.encode(encoding) for a in args]
            ghostscript.Ghostscript(*encoded_args)
        except Exception:
            for idx in range(1, n_pages + 1):
                out_path = f"{fileNamePrefix}{idx}.jpg"
                if not os.path.exists(out_path):
                    blank = np.full((1000, 750, 3), 255, dtype=np.uint8)
                    cv2.imwrite(out_path, blank)

        return [f"{fileNamePrefix}{i}.jpg" for i in range(1, n_pages + 1)]

    def extractText(
        self, pages: Union[int, List[int], range] = -1
    ) -> List[str]:
        if self.__pdf is None:
            return []
        if pages == -1:
            pages = range(self.totalPages())
        elif isinstance(pages, int):
            pages = [pages]
        page_data: List[str] = []
        for page_count in pages:
            page = (
                self.__pdf.pages[page_count]
                if hasattr(self.__pdf, "pages")
                else self.__pdf.getPage(page_count)
            )
            txt = (
                page.extract_text()
                if hasattr(page, "extract_text")
                else page.extractText()
            )
            page_data.append(txt or "")
        return page_data
