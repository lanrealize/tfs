import random

import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from model import TFSModel, train_step, model_evaluate

STATUS_INDICATOR = 1
SEQUENCE_LENGTH = 15
BATCH_SIZE = 32

# Train parameters
EPOCH = 1000
EVALUATE_INTERVAL = 7

data_file = r"./data/train_data.csv"
train_df = pd.read_csv(data_file)

drop_rows = (train_df.loc[train_df.loc[:, 'volume'] == 0]).index.tolist() + [0]
drop_columns = ['date', 'code', 'isST', 'adjustflag', 'tradestatus'] + ['peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']

train_df.drop(drop_columns, axis=1, inplace=True)
train_df.drop(drop_rows, inplace=True)

volume_adjust_factor = 1e7
amount_adjust_factor = 1e8

train_df.loc[:, 'volume'] /= volume_adjust_factor
train_df.loc[:, 'amount'] /= amount_adjust_factor

train_inputs = []
train_targets = []

for idx in tqdm(range(len(train_df))):
    if idx > len(train_df) - 1 - SEQUENCE_LENGTH:
        break
    else:
        train_inputs.append(tf.expand_dims(tf.convert_to_tensor(train_df.iloc[idx: idx + 15], dtype=tf.float32), axis=0))
        train_target = [1, 0, 0] if train_df.iloc[idx + 15, 7] < -STATUS_INDICATOR else [0, 1, 0] if train_df.iloc[idx + 15, 7] < STATUS_INDICATOR else [0, 0, 1]
        train_targets.append(tf.expand_dims(tf.convert_to_tensor(train_target, dtype=tf.int8), axis=0))

train_inputs_tensor = tf.concat(train_inputs, axis=0)
train_targets_tensor = tf.concat(train_targets, axis=0)

sample_length = len(train_df) - SEQUENCE_LENGTH
train_length = int(sample_length * 0.8)
indices = list(range(sample_length))
random.shuffle(indices)

model_train_inputs = tf.gather(train_inputs_tensor, indices[0: train_length], axis=0)
model_val_inputs = tf.gather(train_inputs_tensor, indices[train_length:], axis=0)
model_train_targets = tf.gather(train_targets_tensor, indices[0: train_length], axis=0)
model_val_targets = tf.gather(train_targets_tensor, indices[train_length:], axis=0)

train_ds = tf.data.Dataset.from_tensor_slices({'inputs': model_train_inputs, 'targets': model_train_targets}).shuffle(2048).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices({'inputs': model_val_inputs, 'targets': model_val_targets}).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

model = TFSModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
loss = tf.keras.losses.CategoricalCrossentropy()
loss_metrics = tf.keras.metrics.Mean()

for epoch in range(EPOCH):
    print(f'epoch {epoch}:')
    batch_idx = 0
    for inputs in train_ds:
        train_step(model=model,
                   optimizer=optimizer,
                   loss=loss,
                   loss_metrics=loss_metrics,
                   batched_inputs=inputs['inputs'],
                   batched_targets=inputs['targets'])

        if batch_idx % EVALUATE_INTERVAL == 0:
            model_evaluate(model, val_ds)

        batch_idx += 1
        print(f"train loss is: {loss_metrics.result().numpy()}")

    loss_metrics.reset_states()
