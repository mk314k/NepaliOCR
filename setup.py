from setuptools import find_packages, setup

setup(
    name="nepali_ocr",
    version="0.1.0",
    description="Document layout segmentation and bilingual Devanagari/English OCR pipeline",
    author="Kartikesh Mishra",
    license="MIT",
    packages=find_packages(include=["nepali_ocr", "nepali_ocr.*"]),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "torch>=1.12.0",
        "Pillow>=9.0.0",
        "PyPDF2>=2.0.0",
        "fpdf>=1.7.2",
        "python-docx>=0.8.11",
    ],
    python_requires=">=3.8",
)
