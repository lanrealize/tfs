import tensorflow as tf

lstm = tf.keras.layers.LSTM(20)

inputs = tf.random.normal([32, 10, 8])

output = lstm(inputs)

print()