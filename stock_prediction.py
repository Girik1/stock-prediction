import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Fetch historical stock data
def fetch_stock_data(stock_ticker, start_date, end_date):
    data = yf.download(stock_ticker, start=start_date, end=end_date)
    return data

# Calculate Moving Averages (MA) - Simple Moving Average (SMA)
def calculate_sma(data, window=14):
    sma = data['Close'].rolling(window=window).mean()
    return sma

# Prepare data for LSTM model
def prepare_data_for_lstm(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Scale data between 0 and 1
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM
    return X, y, scaler

# Build the LSTM model
def build_lstm_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))  # Predicting one value (stock price)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the LSTM model
def train_lstm_model(data):
    X, y, scaler = prepare_data_for_lstm(data)
    model = build_lstm_model(X)
    model.fit(X, y, epochs=5, batch_size=32)
    return model, scaler

# Predict stock prices using the trained LSTM model
def predict_stock_price(model, scaler, data, sequence_length=60):
    X, _, _ = prepare_data_for_lstm(data, sequence_length)
    predicted_stock_price = model.predict(X)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)  # Rescale to original price
    return predicted_stock_price

# Plot the actual vs predicted stock prices
def plot_predictions(data, predicted_prices):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Actual Stock Price')
    plt.plot(data.index[sequence_length:], predicted_prices, label='Predicted Stock Price', linestyle='dashed')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Main function to run everything
def main():
    stock_ticker = 'AAPL'  # You can change the stock ticker (e.g., 'GOOG', 'MSFT')
    start_date = '2015-01-01'
    end_date = '2023-01-01'

    # Fetch stock data
    data = fetch_stock_data(stock_ticker, start_date, end_date)

    # Train the model
    model, scaler = train_lstm_model(data)

    # Predict stock prices
    predicted_prices = predict_stock_price(model, scaler, data)

    # Plot the actual vs predicted stock prices
    plot_predictions(data, predicted_prices)

# Run the program
if __name__ == "__main__":
    main()
