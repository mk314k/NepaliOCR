import numpy as np
import cv2
from customDataStruct.pdfClass import PDFReader, PDFWriter
from detector import detect
from recognizer import recognize

def main(fileName):
    pdfIn = PDFReader(fileName, mode='r')
    imgNames=pdfIn.toImage()
    pdfOut = PDFWriter(fileName[:-4]+'-out.pdf',mode='w')

    for imgName in imgNames:
        pdfOut.newPage()
        img = cv2.imread(imgName)
        extractedData = detect(img)

        for dataRect in extractedData:
            if dataRect.type == 'text':
                recognizedText = recognize(dataRect.getImage(img))
                pdfOut.addText(dataRect,recognizedText)
            elif dataRect.type == 'Image':
                pdfOut.addImage(dataRect)
            elif dataRect.type == 'Table':
                pdfOut.addTable(dataRect)
    pass

if __name__=='__main__':
    filename = '' #TODO UI to get filename
    main(filename)
