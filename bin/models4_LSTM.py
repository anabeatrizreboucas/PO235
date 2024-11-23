import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


### CSV file
file_path_pattern = 'yahoo_data/yahoo_data*.csv'
file_list = glob.glob(file_path_pattern)
if file_list:
    file_path = file_list[-1]
    stocks = pd.read_csv(file_path)
    print(stocks.tail())

else:
    print("No files found matching the pattern.")

### Select data
ticker_choosed = 'VALE3'
print(f'Stock: {ticker_choosed}')

data = stocks[stocks['stock'] == ticker_choosed]

dates = pd.to_datetime(data['date']).iloc[:-1].dt.date
data = data[['Adjusted']]

### Create Y
data['next_day_price'] = data['Adjusted'].shift(-1)
data = data.dropna()


### Scale data
train_data_length = int(len(data)*0.8)

train_scaler = MinMaxScaler(feature_range=(0,1))
test_scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = train_scaler.fit_transform(data.iloc[0:train_data_length , :])
test_scaled = test_scaler.fit_transform(data.iloc[train_data_length: , :])               

x_train = train_scaled[:, 0]
y_train = train_scaled[:, 1]
x_test = test_scaled[:, 0]
y_test = test_scaled[:, 1]

#Reshape 3D
x_train = x_train.reshape(x_train.shape[0], 1, 1)
y_train = y_train.reshape(y_train.shape[0], 1, 1)
x_test = x_test.reshape(x_test.shape[0], 1, 1)
y_test = y_test.reshape(y_test.shape[0], 1, 1)

### Model
inputs = keras.layers.Input(shape=(x_train.shape[1], 1)) #1 "X" e 1 output pra esse X. Se também utilizar volume, por exemplo, seria 2.
x = keras.layers.LSTM(50, return_sequences=True)(inputs)
X = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(100, return_sequences=True)(x)
X = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150, return_sequences=True)(x)
X = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(20, activation='linear')(x)
outputs = keras.layers.Dense(1, activation='linear')(x)

neural_net = keras.Model(inputs=inputs, outputs=outputs)
neural_net.compile(optimizer='adam', loss='mse')
neural_net.summary()

### Train
neural_net.fit(
    x_train,
    y_train,
    epochs=80,
    batch_size=8,
    validation_split=0.2
)

### Predict
predicted_prices = neural_net.predict(x_test)
predicted_prices = predicted_prices.reshape(predicted_prices.shape[0], 1)

### Reshape
x_train = x_train.reshape(x_train.shape[0], 1)
y_train = y_train.reshape(y_train.shape[0], 1)
x_test = x_test.reshape(x_test.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

### Join data
data_test = np.concatenate((x_test, y_test), axis=1)
data_predicted = np.concatenate((x_test, predicted_prices), axis=1)

### Inverse scale
test_real_prices = test_scaler.inverse_transform(data_test)
test_predicted_prices = test_scaler.inverse_transform(data_predicted)
pd.DataFrame(test_predicted_prices)

### Plot
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates.iloc[train_data_length:], test_real_prices[:, 1], color='blue', label='Real Price')
ax.plot(dates.iloc[train_data_length:], test_predicted_prices[:, 1], color='red', label='Predicted Price')
plt.title(F'Stock Price Prediction - {ticker_choosed}')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

### Model Evaluation
df = pd.DataFrame(test_predicted_prices, index=dates.iloc[train_data_length:])
df.columns = ['Price', 'Next_day_price_predicted']
df['Results'] = df['Price'].pct_change()
df['Buy_sell'] = pd.NA

df.loc[df['Next_day_price_predicted'] > df['Price'], 'Buy_sell'] = 'Buy'
df.loc[df['Next_day_price_predicted'] < df['Price'], 'Buy_sell'] = 'Sell'

df['Earned'] = pd.NA

df.loc[(df['Buy_sell'] == 'Buy') & (df['Results'] > 0), 'Earned'] = 1
df.loc[(df['Buy_sell'] == 'Buy') & (df['Results'] < 0), 'Earned'] = 0
df.loc[(df['Buy_sell'] == 'Sell') & (df['Results'] > 0), 'Earned'] = 0
df.loc[(df['Buy_sell'] == 'Sell') & (df['Results'] < 0), 'Earned'] = 1
df.loc[df['Earned'].isna(), 'Earned'] = 0

df = df.dropna()
print(df)

right_decisions = df['Earned'].sum()/len(df)
print(f'Right decisions: {right_decisions*100:.2f}%')
wrong_decisions = 1 - right_decisions
print(f'Wrong decisions: {wrong_decisions*100:.2f}%')

df['absolute_results'] = df['Results'].abs()

earns_and_losses_mean = df.groupby('Earned')['absolute_results'].mean()
print(earns_and_losses_mean)

math_expectation = (earns_and_losses_mean[1] * wrong_decisions) - (earns_and_losses_mean[0] * right_decisions)
math_expectation = math_expectation * 100
print(f'Math expectation: {math_expectation:.2f}%')