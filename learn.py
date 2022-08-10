import tensorflow as tf2
import cv2
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Reshape, Bidirectional, LSTM, Dense, Lambda, BatchNormalization
from keras.optimizers import Adam
from utility import unicodes
from keras.applications import ResNet50V2

def ctcLambdaFunc(args):
    """AI is creating summary for ctcLambdaFunc

    Args:
        args ([type]): [description]

    Returns:
        function : [description]
    """
    yPred, labels, inputLength, labelLength = args
    yPred = yPred[:,2:,:]
    return K.ctc_batch_cost(labels,yPred,inputLength,labelLength)

class CRNN():
    """AI is creating summary for CRNN
    """
    def __init__(self, xShape=(32,128,1)) -> None:
        """AI is creating summary for __init__

        Args:
            xShape (tuple, optional): [description]. Defaults to (32,128,1).
        """
        self.__inputX = Input(name = 'inputX', shape=xShape, dtype = 'float32')
        base_model = ResNet50V2(weights=None, input_tensor=self.__inputX, classes=256)
        inner = Reshape(target_shape = (32,256), name = 'reshape')(base_model)
        blstm1 = Bidirectional(LSTM(256, return_sequences = True, kernel_initializer = 'he_normal'))(inner)
        blstm1 = BatchNormalization()(blstm1)
        blstm2 = Bidirectional(LSTM(256, return_sequences = True, kernel_initializer = 'he_normal'))(blstm1)
        blstm2 = BatchNormalization()(blstm2)
        self.__yPred = Dense(len(unicodes)+1, kernel_initializer = 'he_normal', activation = 'softmax')(blstm2)
        self.__labels = Input(name='label', shape=[xShape[0]], dtype='float32')
        self.__inputLength = Input(name='inputLen', shape=[1], dtype='int64')
        self.__labelLength = Input(name='labelLen', shape=[1], dtype='int64')
        self.__lossOut = Lambda(ctcLambdaFunc, output_shape=(1,), name='ctc')([self.__yPred, self.__labels, self.__inputLength, self.__labelLength])
        self.__trainingModelSet = False
        self.__predictionModelSet = False

    def train(self,X,Y,modelPath=None):
        """AI is creating summary for train

        Args:
            X ([type]): [description]
            Y ([type]): [description]
        """
        if not self.__trainingModelSet:
            self.__trainableModel = Model(inputs = [self.__inputX, self.__labels, self.__inputLength, self.__labelLength], outputs=[self.__lossOut,self.__yPred])
            if modelPath: self.__trainableModel.load_weights(modelPath)
            self.__trainingModelSet = True
        self.__trainableModel.compile(optimizer=Adam(learning_rate=0.0001,epsilon=1e-9),loss=ctcLambdaFunc,metrics=['accuracy'])
        self.__trainableModel.fit(X,Y,batch_size=32,epochs=1000)
        #TODO please choose batch_size, epochs and other parameters depending on hardawares
        #TODO loss function look for CTC loss
        #TODO print accuracies 

    def loadWeights(self,modelPath:str):
        """AI is creating summary for loadWeights

        Args:
            modelPath (str): filename including full path to store the model weight
        """
        self.__predictionModel = Model(inputs=[self.__inputX], outputs=self.__yPred)
        self.__predictionModel.load_weights(modelPath,by_name=True)
        self.__predictionModelSet = True

    def predict(self,img:np.ndarray):
        """predicts the model output for given binary image
        Note: input image should be preprocessed and of single channel
                output need to be post processed for generating label

        Args:
            img (ndarray): Binary image 
        """
        if not self.__predictionModelSet:
            raise Exception("model weights need to be loaded before prediction")
        else:
            return self.__predictionModel.predict(img)

    def validate(self):

        pass

    def save_model(self,model_name:str):
        """saves the model weights, should be called after training 

        Args:
            model_name (str): filename including full path to store the model weight
        """
        self.__model.save(model_name+".h5")

if __name__=='__main__':
    #sample code to start training CRNN

    xData = 0 #TODO make it numpy array of all the images after preprocess and converting into binary
    yData = 0 #TODO make it numpy array of labels after preprocessing the devanagari texts
    model = CRNN()
    model.train(xData,yData)
    model.save(model_name='') #TODO full file path for model name (matter of choice) import os for help