#!/usr/bin/python3

import pandas as pd
import numpy as np
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def create_test_dataset(X,time_step):
	Xs = []
	for i in range(len(X) - time_step):
		Xs.append(v)

def create_dataset(X, Y, time_step=1):
    Xs, Ys = [], []
    for i in range(len(X) - time_step):
        v = Y[i:(i + time_step)]
        Xs.append(v)
        Ys.append(Y[i + time_step])
    return np.array(Xs), np.array(Ys)

df = pd.read_csv('Data/data.csv')
df['x'] = list(range(1,len(df)+1))
df['y'] = df['receipt_count'].shift(-1)/df['receipt_count']-1

df.dropna(inplace=True)

X = df['x'].values
y = df['y'].values

time_step = 30  # Number of time steps (lag)
X, y = create_dataset(X, y, time_step)

cut = int(round(len(X)*0.8))
X_train = X[:cut]
X_test = X[cut:]
y_train = np.array(y[:cut])
y_test = y[cut:]

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

fitted_model = model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_split=0.2, verbose=2, callbacks=[early_stopping])
val_loss = fitted_model.history['val_loss']
plt.plot(val_loss)
plt.title('Model Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Validation Loss'], loc='upper right')
plt.show()
y_pred = model.predict(X_test)
preds = []
actuals = []
start = df['receipt_count'].values[cut]
for i,p in enumerate(y_pred):
	if preds == []:
		preds.append((1+p)*start)
		actuals.append((1+y_test[i])*start)
	else:
		preds.append((1+p)*preds[-1])
		actuals.append((1+y_test[i])*actuals[-1])
plt.figure(figsize=(10,6))
plt.plot(actuals, label='Actuals')
plt.plot(preds, label='Predictions')
plt.title('Predictions vs Actuals')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.show()
last_pred = y_pred[-1]
num_future_steps = 365
predictions = []
current_input = y[-time_step:]
for i in range(num_future_steps):
	current_prediction = model.predict(np.array(current_input).reshape(time_step,1))
	predictions.append(np.mean(current_prediction[0]))
	current_input = np.append(current_input,predictions[-1])
	current_input = current_input[-time_step:]

preds = [] 
firstPreds = df['receipt_count'].values.tolist()
for i,p in enumerate(predictions):
	if i == 0: preds.append((1+p)*firstPreds[-1])
	else: preds.append((1+p)*preds[-1])
indices = range(len(firstPreds)+len(preds))
plt.figure(figsize=(10,6))
plt.plot(indices[:len(firstPreds)], firstPreds, label='Past')
plt.plot(indices[len(firstPreds):], preds, label='Predicted')
plt.title('Predictions with Past')
plt.xlabel('Days since Jan 1, 2021')
plt.ylabel('Value')
plt.legend()
plt.show()
date_range = pd.date_range(start='2022-01-01', end='2022-12-31')
dump_df = pd.DataFrame(date_range, columns=['date'])
dump_df['month'] = dump_df['date'].dt.month
dump_df['day'] = dump_df['date'].dt.day
dump_df['predicted_values'] = preds
dump_df.to_csv('Data/predicted_data.csv',index=False)


