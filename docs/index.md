# NepaliOCR

Document layout segmentation and bilingual Devanagari/English OCR pipeline for Nepali textbooks.

Initiated by Kartikesh Mishra ([GitHub Repository](https://github.com/mk314k/NepaliOCR)).

---

## Overview

`NepaliOCR` segments document pages into words (`text`), figures (`Image`), and tables (`Table`) using iterative OpenCV contour bridging and 2D spatial-grid bounding-box merging (`RectSet`), then transcribes bilingual Devanagari and English word crops with a PyTorch CRNN + CTC network.

### Pipeline Stages

1. **Layout Segmentation (`nepali_ocr.detector`, `nepali_ocr.spatial`)**:
   - Thresholds grayscale pages and thickens contours over two passes (`contour_correction`) so characters connected by a Devanagari *shirorekha* merge into word-level components.
   - Classifies each region into `'text'`, `'Image'`, or `'Table'` using rectangular fill ratio and quantized HSV color spread (`merge_and_classify`).
   - Merges overlapping non-table boxes inside a `200 x 200` pixel 2D spatial hash grid (`RectSet`).
2. **Sequence Recognition (`nepali_ocr.model`, `nepali_ocr.recognizer`)**:
   - Normalizes word crops to height `32` with percentile contrast stretching (`preprocess_crop`).
   - Encodes visual features with a residual CNN that preserves `4` vertical bands (`H = 4`) before a 2-layer bidirectional LSTM (`hidden_size = 256`) and greedy CTC decoder (`ctc_greedy_decode`).
3. **Preeti-to-Unicode Conversion (`nepali_ocr.utils`)**:
   - Converts legacy Preeti-encoded `.docx` and PDF text streams into Unicode Devanagari (`preeti_to_unicode`) for ground-truth alignment in `tools/dataset_annotator.py`.

---

## Quick Start

```bash
pip install -e .
python3 -m nepali_ocr.pipeline assets/gr9sctrain23_detected.jpg
```

```python
import cv2
from nepali_ocr import process_image_page

img = cv2.imread("assets/gr9sctrain23_detected.jpg")
result = process_image_page(img)
for item in result["text"][:10]:
    print(item["rect"], "->", item["text"])
```
