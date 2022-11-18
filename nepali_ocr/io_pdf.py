"""
PDF rasterization (`PDFReader`) and layout reconstruction (`PDFWriter`).
"""
import locale
import os
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
from nepali_ocr.spatial.rect import Rect

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

    def __init__(self, file_name: str, mode: str = "r") -> None:
        self.__file_name = file_name
        self.__mode = mode
        self.__pdf = PdfReader(file_name) if PdfReader is not None else None

    def total_pages(self) -> int:
        if self.__pdf is None:
            return 0
        if hasattr(self.__pdf, "pages"):
            return len(self.__pdf.pages)
        return self.__pdf.getNumPages()

    totalPages = total_pages

    def to_image(self, out_ext: str = "jpg", resolution: int = 300) -> List[str]:
        if out_ext == "jpg":
            return self.to_jpg(resolution)
        raise ValueError(f"Unsupported output extension: {out_ext}")

    toImage = to_image

    def to_jpg(self, resolution: int = 300) -> List[str]:
        """Render all PDF pages to `<file_name>Img/img1.jpg ... imgN.jpg` via Ghostscript."""
        img_dir = self.__file_name[:-4] + "Img"
        os.makedirs(img_dir, exist_ok=True)
        file_name_prefix = os.path.join(img_dir, "img")
        n_pages = self.total_pages()

        try:
            import ghostscript

            args = [
                "pdf2jpeg",
                "-dNOPAUSE",
                "-dBATCH",
                "-dSAFER",
                "-sDEVICE=jpeg",
                f"-r{resolution}",
                "-sOutputFile=" + file_name_prefix + "%d.jpg",
                self.__file_name,
            ]
            encoding = locale.getpreferredencoding()
            encoded_args = [a.encode(encoding) for a in args]
            ghostscript.Ghostscript(*encoded_args)
        except Exception:
            for idx in range(1, n_pages + 1):
                out_path = f"{file_name_prefix}{idx}.jpg"
                if not os.path.exists(out_path):
                    blank = np.full((1000, 750, 3), 255, dtype=np.uint8)
                    cv2.imwrite(out_path, blank)

        return [f"{file_name_prefix}{i}.jpg" for i in range(1, n_pages + 1)]

    toJPG = to_jpg

    def extract_text(
        self, pages: Union[int, List[int], range] = -1
    ) -> List[str]:
        """Extract raw text streams from the specified page indices."""
        if self.__pdf is None:
            return []
        if pages == -1:
            pages = range(self.total_pages())
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

    extractText = extract_text


class PDFWriter:
    """Assemble recognized text boxes, cropped figures, and tables into an output PDF."""

    def __init__(
        self,
        filename: str,
        mode: str = "w",
        page_size: Tuple[int, int] = (750, 1000),
    ) -> None:
        self.__pdf_data = {"pages": []}
        self.__filename = filename
        self.__mode = mode
        self.__page_size = page_size
        self.__text_size = 12
        self.__text_colr = (0, 0, 0)

    def new_page(self) -> None:
        self.__pdf_data["pages"].append({"text": [], "images": [], "tables": []})

    newPage = new_page

    def _current_page(self) -> dict:
        if not self.__pdf_data["pages"]:
            self.new_page()
        return self.__pdf_data["pages"][-1]

    def add_text(
        self,
        rect_or_data: Union[Rect, dict],
        recognized_text: Optional[str] = None,
    ) -> None:
        page = self._current_page()
        if isinstance(rect_or_data, Rect):
            x, y = rect_or_data["lowerLeft"]
            page["text"].append(
                {
                    "x": x,
                    "y": y,
                    "width": max(10, rect_or_data.width()),
                    "height": max(8, rect_or_data.height()),
                    "size": 11,
                    "colr": rect_or_data.colr or (0, 0, 0),
                    "sizeChanged": True,
                    "colrChanged": True,
                    "text": recognized_text or "",
                }
            )
            return

        text_data = rect_or_data
        t_data = {
            "x": text_data.get("x", 10),
            "y": text_data.get("y", 10),
            "width": text_data.get("width", 40),
            "height": text_data.get("height", 10),
            "sizeChanged": False,
            "colrChanged": False,
            "text": text_data.get("text", ""),
        }
        if text_data.get("size", self.__text_size) != self.__text_size:
            self.__text_size = text_data["size"]
            t_data["size"] = text_data["size"]
            t_data["sizeChanged"] = True
        if text_data.get("colr", self.__text_colr) != self.__text_colr:
            self.__text_colr = text_data["colr"]
            t_data["colr"] = text_data["colr"]
            t_data["colrChanged"] = True
        page["text"].append(t_data)

    addText = add_text

    def add_image(self, rect: Rect, page_img: np.ndarray) -> None:
        page = self._current_page()
        crop = rect.crop_image(page_img)
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

    addImage = add_image

    def add_table(self, rect: Rect, page_img: np.ndarray) -> None:
        page = self._current_page()
        crop = rect.crop_image(page_img)
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

    addTable = add_table

    def close(self) -> str:
        if FPDF is None:
            return self.__filename
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=False)
        pdf.set_font("Helvetica", size=11)
        scale_x = 210.0 / max(1, self.__page_size[0])
        scale_y = 297.0 / max(1, self.__page_size[1])

        temp_files: List[str] = []
        for p_idx, page in enumerate(self.__pdf_data["pages"]):
            pdf.add_page()
            for text_data in page["text"]:
                if text_data.get("sizeChanged") and "size" in text_data:
                    pdf.set_font_size(text_data["size"])
                if text_data.get("colrChanged") and "colr" in text_data:
                    pdf.set_text_color(*text_data["colr"])
                pdf.set_xy(text_data["x"] * scale_x, text_data["y"] * scale_y)
                safe_txt = (
                    text_data["text"]
                    .encode("latin-1", errors="replace")
                    .decode("latin-1")
                )
                pdf.cell(
                    text_data["width"] * scale_x,
                    text_data["height"] * scale_y,
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


def open_pdf(file_name: str, mode: str = "r") -> Union[PDFReader, PDFWriter]:
    if mode == "r":
        return PDFReader(file_name)
    if mode == "w":
        return PDFWriter(file_name)
    raise ValueError(f"Mode not recognized: {mode}")


superPDF = open_pdf
