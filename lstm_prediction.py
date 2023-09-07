# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./data/elspot-prices_2013_hourly_dkk.csv', delimiter=';')
epochs = 50

# Preprocess the data
df = df.drop(['datetime', 'ELE', 'LV', 'LT'], axis=1) # Drop unnecessary columns
df['Start Hour'] = df['Hours'].apply(lambda x: float(x.split('-')[0]))
df.drop('Hours', axis=1, inplace=True)
df = df.fillna(method='ffill') # Fill missing values with the previous value
scaler = MinMaxScaler() # Scale the data to the range [0, 1]
df_scaled = scaler.fit_transform(df)

# Split the data into train and test sets
train_size = int(len(df_scaled) * 0.8)
train = df_scaled[:train_size, :]
test = df_scaled[train_size:, :]                                                                                   

# Define a function to create sequences of data
def create_sequences(data, seq_length):
    num_features = data.shape[1]
    X = np.zeros((len(data) - seq_length, seq_length, num_features))
    y = np.zeros((len(data) - seq_length, num_features))
    for i in range(len(data) - seq_length):
        X[i,:,:] = data[i:i+seq_length, :]
        y[i,:] = data[i+seq_length, :]
    return X, y

# Create sequences of data with a sequence length of 24 (one day)
seq_length = 24
X_train, y_train = create_sequences(train, seq_length)
X_test, y_test = create_sequences(test, seq_length)

def create_model():
    # print('X_train.shape:', X_train.shape)
    # print('X Columns:', df.columns)
    # print('y_train.shape:', y_train.shape)
    # Build and train the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 16)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(16))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=64)

    # Evaluate the model's performance
    train_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)

    print('Train loss:', train_score)
    print('Test loss:', test_score)

    model.save('lstm_model.h5')

    return model

model = load_model('lstm_model.h5')
# model = create_model()
print(df.columns)
# Use the model to make predictions
predictions = model.predict(X_test)

# Invert the scaling of the predictions and the actual prices
predictions = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test)

# Output the predicted prices for the next 24 hours given the last known price at time t
last_known_price = actual_prices[-1, :]
threshold = 200

for i in range(1, 26):
    input_data = np.reshape(last_known_price, (1, 1, 16))
    next_price = model.predict(input_data)
    next_price = scaler.inverse_transform(next_price)

    last_known_price = next_price
    actual_price = actual_prices[-1+i, :]
    charge_rate = ((next_price[0][0]/threshold)*100) if next_price[0][0] < threshold else 0

    print('Predicted price at t+' + str(i) + 'h:', next_price[0][0])
    print('Actual price at t+' + str(i) + 'h:', actual_price[0])
    print(f'Charge rate: {round(charge_rate, 2)}%')
    print('---')

print('Mean Absolute Error:', mean_absolute_error(actual_prices, predictions))

actual_prices = pd.DataFrame(actual_prices[:, 0])
predictions = pd.DataFrame(predictions[:, 0], columns=['Predicted Prices'])
pred_charge = predictions[predictions['Predicted Prices'] < threshold]
pred_normal = predictions[predictions['Predicted Prices'] >= threshold]

# Plot the predicted prices and actual results
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Prices', color='b')
# plt.plot(predictions[:, 0], label='Predicted Prices')
plt.plot(pred_charge, color='g', label='Predicted Prices (Charge)')
plt.plot(pred_normal, color='orange', label='Predicted Prices (Normal)')
plt.axhline(y = threshold, color = 'black', linestyle = '-')
plt.xlabel('Time (Hours from 2013)')
plt.ylabel('Price (DKK/MWh)')
plt.title('Predicted Prices vs Actual Prices')
plt.legend()
plt.show()