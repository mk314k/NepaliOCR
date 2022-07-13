from fpdf import FPDF
from PyPDF2 import PdfReader
from PIL import Image
import cv2
import numpy as np
import os
import ghostscript 
import locale

class PDFReader():
    def __init__(self,fileName:str) -> None:
        self.__fileName=fileName
    
    def toImage(self, outExt='jpg'):
        if outExt=='jpg':
            self.toJPG()

    def toJPG(self):
        args = ["pef2jpeg", # actual value doesn't matter
            "-dNOPAUSE",
            "-sDEVICE=jpeg",
            "-r144",
            "-sOutputFile=" + self.__fileName[:-4]+'%02d.jpg',
            self.__fileName]

        encoding = locale.getpreferredencoding()
        args = [a.encode(encoding) for a in args]

        ghostscript.Ghostscript(*args)

    def extractText(self,pages:int|list|range=-1):
        pdf = PdfReader(self.__fileName)
        if pages==-1:pages=range(pdf.getNumPages())
        for page_count in pages:
            page = pdf.getPage(page_count)
            page_data = page.extractText()
        return page_data


class PDFWriter():
    def __init__(self,filename) -> None:
        self.__pdf =FPDF(orientation='P',unit='mm', format='A4')
        self.__filename = filename
    
    def addText(self,textData):
        self.__pdf.set_font_size(textData['size'])
        self.__pdf.set_text_color(*textData['colr'])
        self.__pdf.cell(textData['width'],textData['height'],textData['text'])

    def addImage(self,imageData):
        img=Image.fromarray(imageData['img'])
        self.__pdf.image(img,x=imageData['x'],y=imageData['y'],w=imageData['w'],h=imageData['h'])

    def addTable(self,tableData):

        pass
    
    def close(self):
        self.__pdf.output(self.__filename)

def superPDF(fileName:str,mode='r')->object:
    if mode =='r':
        return PDFReader(fileName)
    elif mode=='w':
        return PDFWriter(fileName)
    else:
        assert(False,"mode not recognized")



