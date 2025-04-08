# stock-prediction


How the Code Works:
Fetching Data: We download historical stock data using yfinance based on the ticker symbol (e.g., AAPL for Apple).

Data Preprocessing: We use MinMaxScaler to scale the stock prices between 0 and 1 to make the model more stable.

Model Building: We use a simple LSTM network for predicting future stock prices.

Training: The LSTM model is trained on the last 60 days of stock data.

Prediction and Visualization: The model predicts future stock prices, and we plot the predictions alongside actual stock prices for visual comparison.
