import tensorflow as tf


class TFSModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(50)

    def call(self, inputs, training=False):
        pass

    def get_config(self):
        pass
