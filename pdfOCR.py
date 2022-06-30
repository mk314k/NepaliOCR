import numpy as np
import cv2
import os
import ghostscript
import locale
#from pathlib import Path
#path=Path('.').absolute().__str__()
#print(fpath)
filepath = os.getcwd()
#print(filepath)

def pdf2jpeg(pdf_input_path, jpeg_output_path):
    args = ["pef2jpeg", # actual value doesn't matter
            "-dNOPAUSE",
            "-sDEVICE=jpeg",
            "-r144",
            "-sOutputFile=" + jpeg_output_path,
            pdf_input_path]

    encoding = locale.getpreferredencoding()
    args = [a.encode(encoding) for a in args]

    ghostscript.Ghostscript(*args)

#pdf2jpeg(filepath+"/NepaliOCR/NepaliOCR/bhanutest.pdf",filepath+"/out%02d.jpg")

def pdfToImages(filename:str):
    print(os.environ)