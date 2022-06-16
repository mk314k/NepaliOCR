from detector import detector,FILEPATH,FILEPREFIX
import cv2
import numpy as np

def recognizer(filename:str):
    detected = detector(filename)
    detected["file"]=FILEPATH+FILEPREFIX+filename
    return detected
