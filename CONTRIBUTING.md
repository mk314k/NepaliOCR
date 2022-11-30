# Contributing to NepaliOCR

Thank you for your interest in contributing to `NepaliOCR` (initiated in collaboration with [OLE Nepal](https://www.olenepal.org/)).

## Development Setup

1. Clone the repository and install dependencies in editable mode:
   ```bash
   git clone https://github.com/mk314k/NepaliOCR.git
   cd NepaliOCR
   pip install -e .
   ```

2. Run the unit test suite before submitting changes:
   ```bash
   python3 -m unittest discover -s tests -v
   ```

## Areas for Contribution

- **Layout Segmentation (`nepali_ocr/detector.py`, `nepali_ocr/spatial/`)**:
  - Improving robustness on skewed, low-contrast, or camera-captured pages.
  - Refining table cell extraction and multi-column reading order.
- **Devanagari Recognition & Post-Processing (`nepali_ocr/model.py`, `nepali_ocr/postprocess.py`)**:
  - Expanding multi-font training sets using public Nepali textbook PDFs ([CDC Nepal](https://moecdc.gov.np), [E-Pustakalaya](https://pustakalaya.org)).
  - Adding conjunct ligature and dictionary-assisted post-processing rules.
- **Dataset Annotation (`tools/dataset_annotator.py`)**:
  - Enhancing the Tkinter page-alignment tool for faster ground-truth verification.

## Pull Request Guidelines

- Keep pull requests focused on a single bug fix or feature.
- Add or update behavior-oriented unit tests in `tests/test_ocr.py` for any changes to spatial indexing, layout detection, Preeti conversion, or CTC decoding.
- Follow PEP 8 `snake_case` naming conventions across all modules in `nepali_ocr/`.
