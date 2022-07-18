import cv2
import numpy as np
from utility import CRNNModelPath
import tensorflow as tf2
from learn import recognitionModel

def preProcess(img:np.ndarray):

    pass

def postProcess(pred):

    pass


def recognize(img:np.ndarray):
    imgp = preProcess(img)
    recogModel = recognitionModel(training=False)
    recogModel.load_weights(CRNNModelPath)
    pred = recogModel.predict(imgp)

    return postProcess(pred)
