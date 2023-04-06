import tensorflow as tf
import pandas as pd
from tqdm.auto import tqdm


STATUS_INDICATOR = 1
SEQUENCE_LENGTH = 15

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
        train_target = 0 if train_df.iloc[idx + 15, 7] < -STATUS_INDICATOR else 1 if train_df.iloc[idx + 15, 7] < STATUS_INDICATOR else 2
        train_targets.append(tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(train_target, dtype=tf.int8), axis=0), axis=0))

train_inputs_tensor = tf.concat(train_inputs, axis=0)
train_targets_tensor = tf.concat(train_targets, axis=0)

