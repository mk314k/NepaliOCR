from fpdf import FPDF
from PIL import Image
import cv2
import numpy as np
import os
import ghostscript 
import locale

class PDFReader():
    def __init__(self,fileName:str) -> None:
        self.fileName=fileName
    
    def toImage(self, outExt='jpg'):
        if outExt=='jpg':
            self.toJPG()

    def toJPG(self):
        args = ["pef2jpeg", # actual value doesn't matter
            "-dNOPAUSE",
            "-sDEVICE=jpeg",
            "-r144",
            "-sOutputFile=" + self.fileName[:-4]+'%02d.jpg',
            self.fileName]

        encoding = locale.getpreferredencoding()
        args = [a.encode(encoding) for a in args]

        ghostscript.Ghostscript(*args)


class PDFWriter():
    def __init__(self) -> None:
        
        pass


def superPDF(fileName:str,mode='r')->object:
    if mode =='r':
        return PDFReader(fileName)
    elif mode=='w':
        return PDFWriter(fileName)
    else:
        assert(False,"mode not recognized")



