import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, DepthwiseConv2D, MaxPool2D, Add, ZeroPadding2D, BatchNormalization, Dropout
import tf_slim as slim

import collections
import numpy as np


def bottleneck_block_v2(input_tensor: tf.Tensor, c_output: int, strides: int, kernel_size: int = 3 ,t_expand: int = 6, dropout: float = 0.2) -> tf.Tensor:
    d_input = input_tensor.shape[-1]
    if strides == 1:
        x1 = Conv2D(d_input*t_expand, kernel_size = 1, padding = 'same', activation = 'relu6')(input_tensor)
        x2 = DepthwiseConv2D(kernel_size, strides = 1, padding = 'same', activation = 'relu6')(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(dropout)(x2)
        x3 = Conv2D(c_output, kernel_size = 1, padding = 'same', activation = 'linear' )(x2)

        if input_tensor.shape[-1] == c_output:
            out = Add()([input_tensor, x3])
        else: out = x3

    else:
        x1 = Conv2D(d_input * t_expand, 1, padding = 'same', activation = 'relu6')(input_tensor)
        x2 = DepthwiseConv2D(3, strides = strides, padding = 'same', activation = 'relu6')(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(dropout)(x2)
        x3 = Conv2D(c_output, 1, padding = 'same', activation = 'linear')(x2)

        out = x3

    return out

@slim.add_arg_scope
def apply_activation(x, activation_fn=None, name=None):
    return activation_fn(x, name=name) if activation_fn else x
    '''
    The @slim.add_arg_scope decorator allows you to define an "argument scope" for a function. 
    This means that you can set default arguments for several functions within a certain scope, making your code more concise and easier to manage. 
    When applied to the apply_activation function, it enables the function to be used within such scopes.
    # Example of using apply_activation in a Keras model
        inputs = tf.keras.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(32, (5, 5))(inputs)
        x = apply_activation(x, activation_fn=tf.nn.relu)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (5, 5))(x)
        x = apply_activation(x, activation_fn=tf.nn.relu)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = apply_activation(x, activation_fn=tf.nn.relu)
        outputs = tf.keras.layers.Dense(10)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Now you can train the model as usual
        model.fit(train_images, train_labels, epochs=5)
    '''


print('model_utils.py Passed...')