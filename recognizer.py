import cv2
import numpy as np
from utility import CRNNModelPath
import tensorflow as tf2
from learn import CRNN

def preProcess(img:np.ndarray)->np.ndarray:

    pass

def postProcess(pred:np.ndarray)->str:

    pass


def recognize(img:np.ndarray):
    imgp = preProcess(img)
    recogModel = CRNN()
    recogModel.loadWeights(CRNNModelPath)
    pred = recogModel.predict(imgp)
    return postProcess(pred)
