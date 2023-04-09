import tensorflow as tf

# lstm = tf.keras.layers.LSTM(5, return_state=True, return_sequences=True)

# lstm = tf.keras.layers.LSTM(5)  # [32, 5]

# lstm = tf.keras.layers.LSTM(5, return_sequences=True)  # 32 10 5

lstm = tf.keras.layers.LSTM(5, return_state=True)  # [32, 5]

inputs = tf.random.normal([32, 10, 8])

output = lstm(inputs)

print()
