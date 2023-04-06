import tensorflow as tf
import pandas as pd


data_file = r"./data/train_data.csv"
train_df = pd.read_csv(data_file)
drop_row_index = (train_df.loc[train_df.loc[:, 'volume'] == 0]).index.tolist() + [0]

train_df.drop(drop_row_index, inplace=True)

train_df.loc[:, 'pctChg'].describe()


print(tf.config.list_physical_devices('GPU'))
