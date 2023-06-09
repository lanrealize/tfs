import random

import numpy
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

from config import SEQUENCE_LENGTH, STATUS_INDICATOR, BATCH_SIZE


def build_ds():
    data_file = r"./data/train_data.csv"
    train_df = pd.read_csv(data_file)

    add_features(r"./data/sh.000001.上证综合指数.csv", train_df, 10, 1e10, 1e11, 500)

    adjust_pct(train_df, 'open', 'close', 'pctChg')

    train_df['index'] = train_df.apply(
        lambda x: (x.loc['open'] + x.loc['close'] + x.loc['high'] + x.loc['low']) / 4, axis=1)

    drop_rows = (train_df.loc[train_df.loc[:, 'volume'] == 0]).index.tolist() + [0, 1, 2, 3, 4, 5, 6]
    drop_columns = ['date', 'code', 'isST', 'adjustflag', 'tradestatus'] + ['peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ'] + ['open', 'close', 'high', 'low', 'volume', 'amount', 'sh_amount', 'sh_volume']

    train_df.drop(drop_columns, axis=1, inplace=True)
    train_df.drop(drop_rows, inplace=True)

    # volume_adjust_factor = 1e7
    # amount_adjust_factor = 1e8

    # train_df.loc[:, 'volume'] /= volume_adjust_factor
    # train_df.loc[:, 'amount'] /= amount_adjust_factor

    train_inputs = []
    train_targets = []

    for idx in tqdm(range(len(train_df))):
        if idx > len(train_df) - 1 - SEQUENCE_LENGTH:
            break
        else:
            train_inputs.append(tf.expand_dims(tf.convert_to_tensor(train_df.iloc[idx: idx + SEQUENCE_LENGTH], dtype=tf.float32), axis=0))
            train_target = [1, 0, 0] if train_df.iloc[idx + SEQUENCE_LENGTH, 1] < -STATUS_INDICATOR else [0, 1, 0] if train_df.iloc[idx + SEQUENCE_LENGTH, 1] < STATUS_INDICATOR else [0, 0, 1]
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

    train_ds = tf.data.Dataset.from_tensor_slices(
        {'inputs': model_train_inputs, 'targets': model_train_targets}).shuffle(2048).batch(BATCH_SIZE).prefetch(
        tf.data.experimental.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices({'inputs': model_val_inputs, 'targets': model_val_targets}).batch(
        BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds


def add_features(feature_file, arg_train_df, turn_adjust, volume_adjust, amount_adjust, price_adjust):
    feature_df = pd.read_csv(feature_file)

    start_idx = feature_df.loc[feature_df.loc[:, 'date'] == arg_train_df.iloc[0, 0]].index.tolist()[0]
    end_idx = feature_df.loc[feature_df.loc[:, 'date'] == arg_train_df.iloc[-1, 0]].index.tolist()[0]

    feature_df = feature_df.iloc[start_idx: end_idx + 1].reset_index(drop=True)

    feature_df.loc[:, 'volume'] /= volume_adjust
    feature_df.loc[:, 'amount'] /= amount_adjust
    feature_df.loc[:, 'turn'] *= turn_adjust
    feature_df['sh_index'] = feature_df.apply(
        lambda x: (x.loc['open'] + x.loc['close'] + x.loc['high'] + x.loc['low']) / 4, axis=1)
    feature_df.loc[:, 'sh_index'] /= price_adjust

    feature_dates = feature_df.loc[:, 'date']
    train_dates = arg_train_df.loc[:, 'date']
    if False not in (feature_dates == train_dates).unique() and len((feature_dates == train_dates).unique()) == 1:
        arg_train_df.loc[:, 'sh_index'] = feature_df['sh_index']
        arg_train_df.loc[:, 'sh_turn'] = feature_df['turn']
        arg_train_df.loc[:, 'sh_amount'] = feature_df['amount']
        arg_train_df.loc[:, 'sh_volume'] = feature_df['volume']

        adjust_turn(arg_train_df, 'sh_turn', 'sh_volume')


def adjust_turn(df, turn_column, volume_column):
    for idx, row in df.iterrows():
        if row.loc[turn_column] == 0 or row.loc[turn_column] == numpy.NaN:
            last_turn = df.loc[idx - 1, turn_column]
            last_volume = df.loc[idx - 1, volume_column]
            ratio = last_turn / last_volume

            df.loc[idx, turn_column] = ratio * row.loc[volume_column]


def adjust_pct(df, open_column, close_column, pct_column):
    for idx, row in df.iterrows():
        if row.loc[pct_column] == 0:
            df.loc[idx, pct_column] = (df.loc[idx, close_column] - df.loc[idx, open_column]) / df.loc[idx, open_column]


build_ds()
