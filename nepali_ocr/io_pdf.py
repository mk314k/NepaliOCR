"""
PDF rasterization (`PDFReader`) and layout reconstruction (`PDFWriter`).
"""
import locale
import os
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
from customDataStruct.Rect import Rect

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
        """Render all PDF pages to `<fileName>Img/img1.jpg ... imgN.jpg` via Ghostscript."""
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
        """Extract raw text streams from the specified page indices."""
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


class PDFWriter:
    """Assemble recognized text boxes, cropped figures, and tables into an output PDF."""

    def __init__(
        self,
        filename: str,
        mode: str = "w",
        pageSize: Tuple[int, int] = (750, 1000),
    ) -> None:
        self.__pdfData = {"pages": []}
        self.__filename = filename
        self.__mode = mode
        self.__pageSize = pageSize
        self.__textSize = 12
        self.__textColr = (0, 0, 0)

    def newPage(self) -> None:
        self.__pdfData["pages"].append({"text": [], "images": [], "tables": []})

    def _currentPage(self) -> dict:
        if not self.__pdfData["pages"]:
            self.newPage()
        return self.__pdfData["pages"][-1]

    def addText(
        self,
        rectOrData: Union[Rect, dict],
        recognizedText: Optional[str] = None,
    ) -> None:
        page = self._currentPage()
        if isinstance(rectOrData, Rect):
            x, y = rectOrData["lowerLeft"]
            page["text"].append(
                {
                    "x": x,
                    "y": y,
                    "width": max(10, rectOrData.width()),
                    "height": max(8, rectOrData.height()),
                    "size": 11,
                    "colr": rectOrData.colr or (0, 0, 0),
                    "sizeChanged": True,
                    "colrChanged": True,
                    "text": recognizedText or "",
                }
            )
            return

        textData = rectOrData
        tData = {
            "x": textData.get("x", 10),
            "y": textData.get("y", 10),
            "width": textData.get("width", 40),
            "height": textData.get("height", 10),
            "sizeChanged": False,
            "colrChanged": False,
            "text": textData.get("text", ""),
        }
        if textData.get("size", self.__textSize) != self.__textSize:
            self.__textSize = textData["size"]
            tData["size"] = textData["size"]
            tData["sizeChanged"] = True
        if textData.get("colr", self.__textColr) != self.__textColr:
            self.__textColr = textData["colr"]
            tData["colr"] = textData["colr"]
            tData["colrChanged"] = True
        page["text"].append(tData)

    def addImage(self, rect: Rect, pageImg: np.ndarray) -> None:
        page = self._currentPage()
        crop = rect.getImage(pageImg)
        if crop.size > 0:
            x, y = rect["lowerLeft"]
            page["images"].append(
                {
                    "crop": crop,
                    "x": x,
                    "y": y,
                    "w": rect.width(),
                    "h": rect.height(),
                }
            )

    def addTable(self, rect: Rect, pageImg: np.ndarray) -> None:
        page = self._currentPage()
        crop = rect.getImage(pageImg)
        if crop.size > 0:
            x, y = rect["lowerLeft"]
            page["tables"].append(
                {
                    "crop": crop,
                    "x": x,
                    "y": y,
                    "w": rect.width(),
                    "h": rect.height(),
                }
            )

    def close(self) -> str:
        if FPDF is None:
            return self.__filename
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=False)
        pdf.set_font("Helvetica", size=11)
        scale_x = 210.0 / max(1, self.__pageSize[0])
        scale_y = 297.0 / max(1, self.__pageSize[1])

        temp_files: List[str] = []
        for p_idx, page in enumerate(self.__pdfData["pages"]):
            pdf.add_page()
            for textData in page["text"]:
                if textData.get("sizeChanged") and "size" in textData:
                    pdf.set_font_size(textData["size"])
                if textData.get("colrChanged") and "colr" in textData:
                    pdf.set_text_color(*textData["colr"])
                pdf.set_xy(textData["x"] * scale_x, textData["y"] * scale_y)
                safe_txt = (
                    textData["text"]
                    .encode("latin-1", errors="replace")
                    .decode("latin-1")
                )
                pdf.cell(
                    textData["width"] * scale_x,
                    textData["height"] * scale_y,
                    safe_txt,
                )

            for idx, item in enumerate(page["images"] + page["tables"]):
                tmp_path = f"{self.__filename}_tmp_{p_idx}_{idx}.jpg"
                rgb = (
                    cv2.cvtColor(item["crop"], cv2.COLOR_BGR2RGB)
                    if item["crop"].ndim == 3
                    else item["crop"]
                )
                Image.fromarray(rgb).save(tmp_path)
                temp_files.append(tmp_path)
                pdf.image(
                    tmp_path,
                    x=item["x"] * scale_x,
                    y=item["y"] * scale_y,
                    w=max(5.0, item["w"] * scale_x),
                    h=max(5.0, item["h"] * scale_y),
                )

        pdf.output(self.__filename)
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return self.__filename


def superPDF(fileName: str, mode: str = "r") -> Union[PDFReader, PDFWriter]:
    if mode == "r":
        return PDFReader(fileName)
    if mode == "w":
        return PDFWriter(fileName)
    raise ValueError(f"Mode not recognized: {mode}")
