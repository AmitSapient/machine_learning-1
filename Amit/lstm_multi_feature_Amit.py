# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

time_series_number  = 100

ticker = 'MARUTI'
# Importing the training set
dataset_train = pd.read_csv(ticker+'_Train.csv')
#training_set = dataset_train.iloc[:, 1:2].values
training_set = dataset_train[['Open', 'Total Trade Quantity']]
# Feature Scaling
training_set_scaled = sc.fit_transform(training_set)

train_set_rows = len(training_set_scaled)
print('train_set_rows - ' ,train_set_rows)

dataset_test = pd.read_csv(ticker+'_Test.csv')
dataset_test  = dataset_test [['Open', 'Total Trade Quantity']]


real_stock_price = dataset_test.iloc[:, 0:1].values

dataset_test_scaled = sc.transform(dataset_test)
print('test set lenght - ', len(dataset_test_scaled))

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(time_series_number, train_set_rows):
    X_train.append(training_set_scaled[i-time_series_number:i, ])
#    X_train.append(training_set_scaled[i-60:i, 0 ])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

print('Train Data shapes', X_train.shape,'--', y_train.shape)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))

######## TEST Data ##############
# Getting the real stock price of 2017

test_set_rows = len(dataset_test_scaled)
print('test_set_rows - ' ,test_set_rows)

X_test = []
for i in range(time_series_number, test_set_rows):
    print(i)
    X_test.append(dataset_test_scaled[i-time_series_number:i, ])
#    X_test.append(dataset_test[i-60:i, 0])
#remove the rows above time_series_number to match X_test rows
real_stock_price = real_stock_price[time_series_number:]    
X_test = np.array(X_test)
X_test = np.stack(X_test)
print('Test data shapes', X_test.shape)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))


# Part 2 - Building the RNN

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 2)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))
#
## Adding a third LSTM layer and some Dropout regularisation
#regressor.add(LSTM(units = 50, return_sequences = True))
#regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))
#regressor.add(Activation('softmax'))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#regressor.compile(optimizer = 'sgd', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 30, batch_size = 100)



# Part 3 - Making the predictions and visualising the results


predicted_stock_price = regressor.predict(X_test)
#Amit - below is needed because inverse_transform expect 2 features staructure since
# that is how transform was creayed for 2 features
#https://stackoverflow.com/questions/42997228/lstm-keras-error-valueerror-non-broadcastable-output-operand-with-shape-67704

pred_dataset =  np.zeros(shape=(len(predicted_stock_price), 2) )
pred_dataset[:,0] = predicted_stock_price[:,0]

predicted_stock_price = sc.inverse_transform(pred_dataset)[:,0]

#predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

print('\n RMSE - ', rmse)
diff = real_stock_price - predicted_stock_price
diffArray = np.column_stack((real_stock_price , predicted_stock_price, diff))
print("\n diffArray  - \n", diffArray)
np.savetxt('diffArray.csv',diffArray,delimiter=',')
#diffArray.to_csv("diffArray.csv")