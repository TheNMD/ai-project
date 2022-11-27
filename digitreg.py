import tensorflow as tf
import pandas as pd

csv = pd.read_csv('./data/csv/train.csv')
train_image = csv.iloc[0:10000, 1:]
test_image = csv.iloc[10001:20000, 1:]
train_label = csv.iloc[10001:20000, :1]