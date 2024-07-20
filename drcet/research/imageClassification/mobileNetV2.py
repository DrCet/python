
from model_utils import bottleneck_block_v2

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Dropout

import numpy as np
import copy
import functools

class MobileNetV2(tf.keras.Model):
    def __init__(self, n_class: int):
        super().__init__()
        self.n_class = n_class
        self.build_model()


    def build_model(self, input_shape = (224, 224, 3)):
        self.input_ = Input(shape=input_shape)
        
        x = Conv2D(32, kernel_size=3, padding='same', strides=2, activation='relu6')(self.input_)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = bottleneck_block_v2(x, c_output=16, t_expand=1, strides=1)

        x = bottleneck_block_v2(x, c_output=24, strides=2)
        x = bottleneck_block_v2(x, c_output=24, strides=1)

        x = bottleneck_block_v2(x, c_output=32, strides=2)
        x = bottleneck_block_v2(x, c_output=32, strides=1)
        x = bottleneck_block_v2(x, c_output=32, strides=1)

        x = bottleneck_block_v2(x, c_output=64, strides=2)
        x = bottleneck_block_v2(x, c_output=64, strides=1)
        x = bottleneck_block_v2(x, c_output=64, strides=1)
        x = bottleneck_block_v2(x, c_output=64, strides=1)

        x = bottleneck_block_v2(x, c_output=96, strides=1)
        x = bottleneck_block_v2(x, c_output=96, strides=1)
        x = bottleneck_block_v2(x, c_output=96, strides=1)

        x = bottleneck_block_v2(x, c_output=160, strides=2)
        x = bottleneck_block_v2(x, c_output=160, strides=1)
        x = bottleneck_block_v2(x, c_output=160, strides=1)

        x = bottleneck_block_v2(x, c_output=320, strides=1)

        x = Conv2D(1280, kernel_size=1, strides=1, padding='same')(x)
        x = AveragePooling2D((7, 7), strides=1)(x)

        x = Flatten()(x)
        self.output_ = Dense(self.n_class, activation='softmax')(x)
        self.model_ = Model(inputs=self.input_, outputs=self.output_)


    def call(self,inputs):
        x = self.model_(inputs)
        return x
      

    def summary(self):
        self.model_.summary()


    def infer(self, inputs: tf.Tensor, label_map: dict):
        if len(inputs.shape) != 4 or inputs.shape[1:] != (224,224,3):
            raise ValueError('Input must have shape (batch_size, 224, 224, 3)')
        
        predictions = np.argmax(self.model(inputs), axis = -1)
        return [label_map[pred] for pred in predictions]
        
