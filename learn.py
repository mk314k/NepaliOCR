import tensorflow as tf2
import cv2
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout, Add
from keras.optimizers import Adam
from utility import unicodes

def ctcLambdaFunc(args):
    yPred, labels, inputLength, labelLength = args
    yPred = yPred[:,2:,:]
    loss = K.ctc_batch_cost(labels,yPred,inputLength,labelLength)
    return loss

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True, pad = 'same' ):
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding= pad,
                  kernel_initializer='he_normal')

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def res_block(inputs,num_filters=16,kernel_size=3,strides=1,padding = 'same',activation='relu',batch_normalization=True,conv_first=True,BN=True,A=True):
    x = inputs
    y = resnet_layer(inputs=x,num_filters=num_filters,strides=strides, pad = padding)
    y = resnet_layer(inputs=y,num_filters=num_filters,activation=None, pad = padding)
    x = resnet_layer(inputs=x,num_filters=num_filters,strides=strides,pad = padding,activation=None,batch_normalization=False)
    x = Add([x, y])
    if BN:
        x = BatchNormalization()(x)
    if A:
        x = Activation('relu')(x)
    return x

def recognitionModel(training):
    inputs = Input(name = 'inputX', shape=(32,128,1), dtype = 'uint8')
    inner = res_block(inputs,64)
    inner = res_block(inner,64)
    inner = MaxPooling2D(pool_size = (2,2),name = 'MaxPoolName1')(inner)
    inner = res_block(inner,128)
    inner = res_block(inner,128)
    inner = MaxPooling2D(pool_size = (2,2),name = 'MaxPoolName2')(inner)
    inner = res_block(inner,256)
    inner = res_block(inner,256)
    inner = MaxPooling2D(pool_size = (1,2),strides = (2,2), name = 'MaxPoolName4')(inner)
    inner = res_block(inner,512)
    inner = res_block(inner,512)
    inner = MaxPooling2D(pool_size = (1,2), strides = (2,2), name = 'MaxPoolName5')(inner)
    inner = res_block(inner,512)
    inner = Reshape(target_shape = (32,256), name = 'reshape')(inner)
    blstm1 = Bidirectional(LSTM(256, return_sequences = True, kernel_initializer = 'he_normal'))(inner)
    blstm1 = BatchNormalization()(blstm1)
    blstm2 = Bidirectional(LSTM(256, return_sequences = True, kernel_initializer = 'he_normal'))(blstm1)
    blstm2 = BatchNormalization()(blstm2)
    yPred = Dense(len(unicodes)+1, kernel_initializer = 'he_normal', activation = 'softmax')(blstm2)

    labels = Input(name='label', shape=[32], dtype='float32')
    inputLength = Input(name='inputLen', shape=[1], dtype='int64')
    labelLength = Input(name='labelLen', shape=[1], dtype='int64')

    lossOut = Lambda(ctcLambdaFunc, output_shape=(1,), name='ctc')([yPred, labels, inputLength, labelLength])

    if training:
        return Model(inputs = [inputs, labels, inputLength, labelLength], outputs=[lossOut,yPred])
    return Model(inputs=[inputs], outputs=yPred)


class CRNN():
    def __init__(self) -> None:
        self.__model = recognitionModel(training=True)

    def train(self,X,Y):
        self.__model.compile()
        self.__model.fit()
        #TODO compile and fit model

    def validate(self):

        pass

    def save_model(self,model_name:str):
        self.__model.save(model_name+".h5")

