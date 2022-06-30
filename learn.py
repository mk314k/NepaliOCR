import tensorflow as tf2
import cv2
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam


class CRNN():
    def __init__(self) -> None:
        inputx=Input(shape=(128,64,1),name='input')
        inputy=Input()
        inner = Conv2D(32,(2,2),name='conv1')(inputx)
        #TODO Make whole model here

        outy=Activation('softmax',name='softmax')(inner)

        self.model = Model(inputs=[inputx,inputy],outputs=self.loss())
        pass

    def loss(self):
        #TODO Make a loss function
        return

    def train(self,X,Y):
        self.model.compile()
        self.model.fit()
        #TODO compile and fit model

    def validate(self):

        pass

    def save_model(self,model_name:str):
        self.model.save(model_name+".h5")

