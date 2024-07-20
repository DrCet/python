# this is an attempt to build an OCR model with text dectection using EAST approach and CTC model for text regconition
# this file is still on working 

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Conv2D, BatchNormalization, Dropout, AveragePooling2D, Reshape
from tensorflow.keras import Model, Input
import numpy as np


from drcet.research.OCR.losses import CTCloss
from drcet.research.imageClassification.model_utils import bottleneck_block_v2
from tensorflow.keras.applications import EfficientNetB0


class TextDetection(Model):
    def __init__(self, input_shape: tuple = (826,585,3)):
        super().__init__()
        self.input_shape_ = input_shape
        self.backbone = EfficientNetB0(include_top = False, input_shape = self.input_shape_)
        self.build_model()

    def build_model(self):
        self.input_ = Input(shape = self.input_shape_)
        x = self.backbone(self.input_) # self.backbone()(self.input_) is the wrong syntax
        self.output_ = Conv2D(1, (1,1), activation = 'sigmoid')(x)
        self.model_ = Model(self.input_, self.output_)

    def call(self, inputs):
        return self.model_(inputs)
    
    def summary(self):
        return self.model_.summary()

class JulieOCR(Model):
    def __init__(self, n_class: int):
        super().__init__()
        self.n_class = n_class
        self.build_text_recognition()
        
    def build_text_recognition(self, input_shape_= (512, 256, 1)):
        self.input_ = Input(shape=input_shape)
        
        x = Conv2D(32, kernel_size=3, padding='same', strides=2, activation='relu6')(self.input_)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = bottleneck_block_v2(x, c_output=16, t_expand=1, strides=1)

        x = bottleneck_block_v2(x, c_output=24, strides=2)
        x = bottleneck_block_v2(x, c_output=24, strides=1)

        x = bottleneck_block_v2(x, c_output=32, strides=2)
        x = bottleneck_block_v2(x, c_output=32, strides=1)

        x = bottleneck_block_v2(x, c_output=64, strides=2)
        x = bottleneck_block_v2(x, c_output=64, strides=1)

        x = bottleneck_block_v2(x, c_output=96, strides=1)

        x = bottleneck_block_v2(x, c_output=128, strides=2)
        x = bottleneck_block_v2(x, c_output=128, strides=1)

        x = bottleneck_block_v2(x, c_output=160, strides=1)

        x = Conv2D(200, kernel_size=1, strides=1, padding='same')(x)
        x = AveragePooling2D((7, 7), strides=1)(x)

        reshaped = Reshape((x.shape[1]*x.shape[2], x.shape[3]))(x)

        lstm = Bidirectional(LSTM(64, return_sequences = True))(reshaped)
        lstm = Bidirectional(LSTM(64, return_sequences = True))(lstm)

        self.output_ = Dense(self.n_class, activation = 'softmax')(lstm)
        self.model = Model(inputs=self.input_, outputs=self.output_)
    def summary(self):
        return self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

textDetector = TextDetection()
textDetector.summary()

print('julieOCR.py pass')