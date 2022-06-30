import numpy as np
import cv2
import os
from pdfClass import superPDF
from detector import detect
from recognizer import recognize

filepath = os.getcwd()

def main(fileName):
    pdfIn = superPDF(fileName, mode='r')
    imgNames=pdfIn.toImage()
    pdfOut = superPDF(fileName[:-4]+'-out.pdf',mode='w')
    for imgName in imgNames:
        pdfOut.newPage()
        img = cv2.imread(imgName)
        extractedData = detect(img)
        for imgData in extractedData.texts():
            imgBlock = extractedData.getImage(imgData)
            textBlock = recognize(imgBlock)
            pdfOut.addText(textBlock,imgData)

        for imgData in extractedData.images():
            img=extractedData.getImage(imgData)
            pdfOut.addImage(img,imgData)

    pdfOut.close()
    pdfIn.close()

            





    pass

if __name__=='__main__':
    #TODO UI to get filename
    main(filepath+"")