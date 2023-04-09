import tensorflow as tf
import math
import numpy as np


class TFSModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm1 = tf.keras.layers.LSTM(64, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.lstm3 = tf.keras.layers.LSTM(256)
        self.dense = tf.keras.layers.Dense(64)
        self.regressor = tf.keras.layers.Dense(3)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=False):
        lstm1_outputs = self.lstm1(inputs)
        # lstm1_outputs = tf.keras.layers.Dropout(0.3)(lstm1_outputs)
        lstm2_outputs = self.lstm2(lstm1_outputs)
        # lstm2_outputs = tf.keras.layers.Dropout(0.3)(lstm2_outputs)
        lstm3_outputs = self.lstm3(lstm2_outputs)
        # lstm3_outputs = tf.keras.layers.Dropout(0.3)(lstm3_outputs)
        dense_outputs = self.dense(lstm3_outputs)  # [batch_size, 128]
        # dense_outputs = tf.keras.layers.Dropout(0.3)(dense_outputs)
        regressor_outputs = self.regressor(dense_outputs)  # [batch_size, 3]
        # regressor_outputs = tf.keras.layers.Dropout(0.3)(regressor_outputs)
        outputs = self.softmax(regressor_outputs)  # [batch_size, 3]
        return outputs

    def get_config(self):
        pass


@tf.function()
def train_step(model, optimizer, loss, loss_metrics, batched_inputs, batched_targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs=batched_inputs, training=True)
        loss = loss(batched_targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss = tf.keras.losses.mean_squared_error(batched_targets, predictions)
    loss_metrics(train_loss)


def model_evaluate(model, dataset):
    predictions = []
    targets = []

    for inputs in dataset:
        prediction = model(inputs=inputs['inputs'], training=False)

        predictions.append(prediction)
        targets.append(inputs['targets'])

    predictions = tf.concat(predictions, axis=0)
    targets = tf.concat(targets, axis=0)

    evaluate_loss = np.mean(tf.keras.losses.mean_squared_error(targets, predictions).numpy())

    print(f'evaluate loss is {math.sqrt(evaluate_loss):.5f}')
