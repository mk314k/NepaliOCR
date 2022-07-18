import numpy as np
from detector import customDetector
from utility import showImage, preeti2Unicode
import cv2
from pdfClass import PDFReader

detectFunc = customDetector()
dataSavePath =""

def showAllDetectedRegion(imgo:np.ndarray, imageTitle:str, showLabel=False, source=None, saveOption = 'Image' ):
    img = np.array(imgo)
    label = 1
    result = {}
    detectedRects = detectFunc(img)
    for rect in detectedRects:
        cv2.rectangle(img,rect["lowerLeft"], rect["upperRight"], (0,0,255), 1)
        if source:
            if saveOption == 'Image':
                pt1 = rect['lowerLeft']
                pt2 = rect['upperRight']
                imgs = imgo[pt1[0]:pt2[0], pt1[1]:pt2[1]]
                cv2.imwrite(source[:-4]+f'{label}.jpg',imgs)
            elif saveOption == 'file':
                result[label] = rect
        if showLabel:
            cv2.putText(img,f'{label}', rect['lowerLeft'],color=(255,0,0),thickness=1)
        label += 1
    showImage(imageTitle, img)
    return result

def main(filename:str, imageRange = None, saveOption ='file', textConversion = False):
    if not imageRange:
        pdf = PDFReader(filename)
        imgNames = pdf.toImage()
        allTexts = pdf.extractText()
        if textConversion:
            allTexts = preeti2Unicode(allTexts)
        with open (filename[:-4]+'.txt', 'w') as file:
            file.write(allTexts)
    else:
        imgNames = [filename[:-4]+f'{i}.jpg' for i in range(imageRange[0],imageRange[1])]

    for imgPath in imgNames:
        imgo = cv2.imread(imgPath)
        result = showAllDetectedRegion(imgo,"Without Labels", source=imgPath, saveOption=saveOption)
        showAllDetectedRegion(imgo,"With Labels", True)
        if saveOption == 'file':
            with open (imgPath[:-4]+'.txt','w') as rectFile:
                rectFile.write(result.__str__())

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    fileName = ''
    main(fileName)

    
    


