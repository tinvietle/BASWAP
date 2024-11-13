import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from pandas import concat
from numpy import concatenate
import joblib
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import os
import logging
import logging.handlers
import requests
import json
import csv
import pytz
from datetime import date, timedelta, datetime
from datahub import sendMessage
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger_file_handler = logging.handlers.RotatingFileHandler(
    "status.log",
    maxBytes=1024 * 1024,
    backupCount=1,
    encoding="utf8",
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger_file_handler.setFormatter(formatter)
logger.addHandler(logger_file_handler)

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)

import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
keras.utils.set_random_seed(812)

try:
    USERNAME_DATAHUB = os.environ["USERNAME_DATAHUB"]
    PASSWORD_DATAHUB = os.environ["PASSWORD_DATAHUB"]
except KeyError:
    logger.info("Environment variables not set!")
    #raise

train_year = [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011]
test_year = [2012, 2013, 2014]
# stations = ["TramTraVinh", "TramCauQuanDinh"]
stations = ["TramTraVinh"]

train_data = []
test_data = []

#TramTraVinh/NoEEMD

for station in stations:
    data = pd.read_csv(f'{station}/NoEEMD/{station}NoEEMD.csv', header=0, index_col=0).dropna()
    train_data.append(data.loc[data["Year"].isin(train_year)])
    test_data.append(data.loc[data["Year"].isin(test_year)])

train_data = pd.concat(train_data)
test_data = pd.concat(test_data)

# load dataset
train_values = train_data.values
test_values = test_data.values

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, input_gap=False):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	if input_gap:
		cols.append(df.shift(n_in))
		names += [('var%d(t-%d)' % (j+1, n_in)) for j in range(n_vars)]
	else:
		for i in range(n_in, 0, -1):
			cols.append(df.shift(i))
			names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
train_values = train_data.iloc[:, 3:].values.reshape(-1, train_data.shape[1]-3)
test_values = test_data.iloc[:, 3:].values.reshape(-1, test_data.shape[1]-3)
# # integer encode direction
# encoder = LabelEncoder()
# values[:,1] = encoder.fit_transform(values[:,1])
# ensure all data is float
train_values = train_values.astype('float32')
test_values = test_values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.vstack((train_values[:, -1].reshape(-1, 1), test_values[:, -1].reshape(-1, 1))))
joblib.dump(scaler, 'scaler.pkl')
train_scaled = np.concatenate([train_values[:, :-1], scaler.transform(train_values[:, -1].reshape(-1, 1))], axis=1)
test_scaled = np.concatenate([test_values[:, :-1], scaler.transform(test_values[:, -1].reshape(-1, 1))], axis=1)
# frame as supervised learning
lookback = 30
day_ahead = 1
train_reframed = series_to_supervised(train_scaled, lookback, day_ahead, input_gap=False)
test_reframed = series_to_supervised(test_scaled, lookback, day_ahead, input_gap=False)

train = train_reframed.values
test = test_reframed.values
# split into input and outputs
train_X, train_y = train[:, :-day_ahead], train[:, -day_ahead:]
test_X, test_y = test[:, :-day_ahead], test[:, -day_ahead:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

scaler = joblib.load('scaler.pkl')

def invert_scaling(y_pred, y_test):
    # invert scaling for forecast
    # inv_y_pred = concatenate((y_pred, test_X[:, -(train_values.shape[1]-1):]), axis=1)
    # inv_y_pred = scaler.inverse_transform(inv_y_pred)
    inv_y_pred = scaler.inverse_transform(y_pred)
    # inv_y_pred = y_pred
    inv_y_pred = inv_y_pred
    # invert scaling for actual
    # inv_y_gt = concatenate((test_y, test_X[:, -(test_values.shape[1]-1):]), axis=1)
    # inv_y_gt = scaler.inverse_transform(inv_y_gt)
    inv_y_gt = scaler.inverse_transform(test_y)
    # inv_y_gt = test_y
    inv_y_gt = inv_y_gt
    return inv_y_pred, inv_y_gt

model = keras.models.load_model("my_best_model_lord.keras")

# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# test_X = scaler.inverse_transform(test_X)
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

y_pred = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))
test_y = test_y.reshape((len(test_y), day_ahead))
y_pred = y_pred.reshape((len(y_pred), day_ahead))
inv_y_pred, inv_y_gt = invert_scaling(y_pred, test_y)
print("The next EC value is: ", round(inv_y_pred[0, 0]))
logger.info(f'The next EC value is: {round(inv_y_pred[0, 0])}')
sendMessage(USERNAME_DATAHUB, PASSWORD_DATAHUB, "Predicted_EC_Value", round(inv_y_pred[0, 0]))

# Set up a 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
fig.suptitle('Line Plots for Each Column')

# Plot each column in a separate subplot
for i in range(inv_y_gt.shape[1]):
    mae = mean_absolute_error(inv_y_gt[:, i], inv_y_pred[:, i])
    mse = mean_squared_error(inv_y_gt[:, i], inv_y_pred[:, i])
    r_squared = sklearn.metrics.r2_score(inv_y_gt[:, i], inv_y_pred[:, i])
    print(f"Time step: {i}")
    print(f'- Test MAE: {mae:.3f}')
    print(f'- Test MSE: {mse:.3f}')
    print(f'- Test R^2: {r_squared:.3f}')
    row = i // 3  # Determine row index
    col = i % 3   # Determine column index
    axes[row, col].plot(inv_y_gt[:, i])
    axes[row, col].plot(inv_y_pred[:, i])
    axes[row, col].set_title(f'Column {i+1}')
    axes[row, col].set_xlabel('Index')
    axes[row, col].set_ylabel('Value')
    logger.info(f'MAE: {mae:.3f}, MSE: {mse:.3f}, R^2: {r_squared:.3f}')

# Hide the last empty subplot (bottom-right)
axes[2, 2].axis('off')


# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()