import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.backend import ctc_batch_cost

class CTCloss(Loss):
    def __init__(self, name: str = 'CTCloss'):
        super().__init__(name = name)
        self.loss_fn = ctc_batch_cost
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        batch_size = tf.cast(tf.shape(y_true)[0], dtype = 'int64')

        input_length = tf.cast(tf.shape(y_pred)[1], dtype = 'int64')
        input_length = input_length * tf.ones(shape = (batch_size, 1), dtype = 'int64')

        label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), dtype = 'int64'), axis = 1)
        label_length = tf.cast(tf.expand_dims(label_length, axis = 1), dtype = 'int64')

        cost = self.loss_fn(y_true, y_pred, input_length, label_length)
        return cost

print('losses.py passed...')