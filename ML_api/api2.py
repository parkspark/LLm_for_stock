import numpy as np
from sklearn.linear_model import LinearRegression
import psycopg2
from flask import Flask, request, jsonify
app = Flask(__name__)

## add by JIN
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@app.route('/')
def welcome():
  return 'HELLO, ML API SERVER'

@app.route('/predict', methods=['POST'])
def predict():
  '''
  data = request.json
  new_X = data['input']
  new_X_val = float(new_X[0])
  input_X = np.array(new_X_val).reshape(1, -1)
  y_pred = model.predict(input_X)
  y_pred_list = y_pred.tolist()
  y_pred_val = round(y_pred_list[0][0], 5)
  '''

  tf.random.set_seed(10) ##

  ### load data
  dir = 'time_series_top5/'
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

  reconstructed_model = tf.keras.models.load_model('simple_lstm_model')

  # to inference
  for new_X_val, y in val_univariate.take(1):
    y_pred_val = reconstructed_model.predict(x)[0])


  conn = psycopg2.connect(dbname='ml', user='postgres', password='postgres', host='127.0.0.1', port=5432)
  cur = conn.cursor()
  query = "INSERT INTO pred_result (input, output) VALUES (%s, %s)"
  values = (new_X_val, y_pred_val)
  cur.execute(query, values)
  conn.commit()
  cur.close()
  conn.close()
  
  res = jsonify({'input': new_X_val, 'predicted_output': y_pred_val})
  return res

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8989)
