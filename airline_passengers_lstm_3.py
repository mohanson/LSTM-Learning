# LSTM for Regression with Time Steps
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sklearn
import sklearn.preprocessing
import sklearn.metrics
import tensorflow as tf

c_csv_file = 'res/immortal_experiment/airline_passengers.csv'

# Fix random seed for reproducibility
np.random.seed(7)

dataframe = pandas.read_csv(c_csv_file, usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
# Normalize the dataset
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]


def create_dataset(dataset, look_back=1):
    # Convert an array of values into a dataset matrix
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# Reshape into X=t and Y=t+1
look_back = 3
train_x, train_y = create_dataset(train, look_back)
test_x, test_y = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

# Create and fit the LSTM network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(4, input_shape=(look_back, 1)))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_x, train_y, epochs=100, batch_size=1, verbose=2)

# Make predictions
train_predict = model.predict(train_x)
test_predict = model.predict(test_x)
# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
train_y = scaler.inverse_transform([train_y])
test_predict = scaler.inverse_transform(test_predict)
test_y = scaler.inverse_transform([test_y])

# Calculate root mean squared error
train_score = math.sqrt(sklearn.metrics.mean_squared_error(train_y[0], train_predict[:, 0]))
print('Train Score: %.2f RMSE' % (train_score))
test_score = math.sqrt(sklearn.metrics.mean_squared_error(test_y[0], test_predict[:, 0]))
print('Test Score: %.2f RMSE' % (test_score))

# Shift train predictions for plotting
train_predictPlot = np.empty_like(dataset)
train_predictPlot[:, :] = np.nan
train_predictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# Shift test predictions for plotting
test_predictPlot = np.empty_like(dataset)
test_predictPlot[:, :] = np.nan
test_predictPlot[len(train_predict)+(look_back*2)+1:len(dataset)-1, :] = test_predict

# Plot baseline and predictions
plt.figure(figsize=(19.2, 10.8))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(train_predictPlot)
plt.plot(test_predictPlot)
plt.savefig('/tmp/out.png')
