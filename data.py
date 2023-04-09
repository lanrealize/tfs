import random

import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm


SEQUENCE_LENGTH = 20
STATUS_INDICATOR = 1
BATCH_SIZE = 64

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