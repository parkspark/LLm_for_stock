import os
import pandas as pd
import tensorflow as tf
import numpy as np


tf.random.set_seed(10) ##

### load data
dir = 'for_lstm_api/time_series_top5/'
df = pd.read_excel(dir + 'samsung_ts_preprocessed3.xlsx')

### split train and test data
test_sd = 20200303
train = df.loc[df['date'] < test_sd]
test = df.loc[df['date'] >= test_sd]

### scale for train data
target = 'foreign_volume'
abs_max = train[target].abs().max() # 1121643
train['scaled'] = train[target] / abs_max
test['scaled'] = test[target] / abs_max

# Define a specific window for training Neural Network
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

df2 = pd.concat([train, test])

uni_data = df2['scaled'].values
TRAIN_SPLIT = len(train) #4782

univariate_past_history = 60 # 30 days
univariate_future_target = 0

x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                     univariate_past_history,
                                     univariate_future_target)

BATCH_SIZE = 128
BUFFER_SIZE = 1000

val_steps = len(x_val_uni) / BATCH_SIZE
print('val_steps :', val_steps, int(val_steps))

# 학습데이터 제공 파이프라인
val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

reconstructed_model = tf.keras.models.load_model('for_lstm_api/simple_lstm_model')

# to inference
cnt = 0
for x, y in val_univariate.take(1):
    print(reconstructed_model.predict(x)[0])
    
# [0.02731112] 보정값을 곱해주면 외국인 거래량 이 됨
# 