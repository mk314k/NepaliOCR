from fpdf import FPDF
from PyPDF2 import PdfReader
from PIL import Image
import cv2
import numpy as np
import ghostscript 
import locale
import os

class PDFReader():
    def __init__(self,fileName:str) -> None:
        self.__fileName=fileName
        self.__pdf = PdfReader(fileName)

    def totalPages(self):
        return self.__pdf.getNumPages()
    
    def toImage(self, outExt='jpg',resolution = 300):
        if outExt=='jpg':
            return self.toJPG(resolution)

    def toJPG(self, resolution=300):
        fileNameOnly = self.__fileName[:-4]+'Img/img'
        os.mkdir(fileNameOnly[:-4])
        args = ["pef2jpeg", # actual value doesn't matter
            "-dNOPAUSE",
            "-sDEVICE=jpeg",
            f"-r{resolution}",
            "-sOutputFile=" +fileNameOnly+'%d.jpg',
            self.__fileName]

        encoding = locale.getpreferredencoding()
        args = [a.encode(encoding) for a in args]
        ghostscript.Ghostscript(*args)

        return fileNameOnly+'.jpg'

    def extractText(self,pages:int|list|range=-1):
        pdf = PdfReader(self.__fileName)
        if pages==-1:pages=range(pdf.getNumPages())
        page_data =[]
        for page_count in pages:
            page = pdf.getPage(page_count)
            page_data.append(page.extractText())
        return page_data


class PDFWriter():
    def __init__(self,filename) -> None:
        self.__pdfData= {'text':[]}
        self.__filename = filename
        self.__textSize = 0
        self.__textColr = (255,255,255)
    
    def addText(self,textData):
        tData = {}
        tData['sizeChanged'] = False
        tData['colorChanged'] = False
        if textData['size'] != self.__textSize:
            self.__textSize = textData['size']
            tData['size'] = textData['size']
            tData['sizeChanged'] = True
        if textData['colr'] != self.__textColr:
            self.__textColr = textData['colr']
            tData['colr'] = textData['colr']
            tData['colrChanged'] = True
        tData['text'] = textData['text']

    # def addImage(self,imageData):
    #     img=Image.fromarray(imageData['img'])
    #     self.__pdf.image(img,x=imageData['x'],y=imageData['y'],w=imageData['w'],h=imageData['h'])

    # def addTable(self,tableData):

    #     pass
    
    def close(self):
        pdf =FPDF(orientation='P',unit='mm', format='A4')
        for textData in self.__pdfData['text']:
            if textData['sizeChanged']: pdf.set_font_size(textData['size'])
            if textData['colrChanged']: pdf.set_text_color(*textData['colr'])
            pdf.cell(textData['width'],textData['height'],textData['text'])

        pdf.output(self.__filename)

def superPDF(fileName:str,mode='r')->object:
    if mode =='r':
        return PDFReader(fileName)
    elif mode=='w':
        return PDFWriter(fileName)
    else:
        assert(False,"mode not recognized")



