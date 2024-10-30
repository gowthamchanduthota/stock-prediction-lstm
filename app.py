import streamlit as st
import pandas as pd
import numpy as np
import pytz
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime
import math
from sklearn.metrics import mean_squared_error


# %matplotlib inline
# plt.plot(df2['Close'])
# plt.show()

def plot_graph(figsize, values, col):
    plt.figure(figsize = figsize)
    values.plot()
    plt.xlabel("Years")
    plt.ylabel(col)
    plt.title("Stock Data Prediction")

st.title("Stock Predictions App")
stock_name = st.text_input("Enter Stock name", "AAPL")
# stock_name = "AAPL"
att_name = st.text_input("Enter the attribute to predict", "Close")
# att_name = "Close"
epochs = st.number_input("Optional: Specify no. of Epochs", 10)


tz = pytz.timezone("America/New_York")
end = tz.localize(datetime.today())
# start = tz.localize(dt(2013,1,1))
# end = datetime.now()
start = datetime(end.year-5, end.month, end.day)


# Downloading data
stock_data = yf.download(stock_name, start=start, end=end)
st.subheader("Stock Data")
st.write(stock_data)


# Preprocessing
stock_data_att = stock_data[[att_name]]
print(len(stock_data_att))

st.pyplot(plot_graph((15, 5), stock_data[att_name], att_name))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data_att)

X_data = []
y_data = []

past_mem = 100

for i in range(past_mem, len(scaled_data)):
    X_data.append(scaled_data[i-past_mem:i])
    y_data.append(scaled_data[i])

X_data = np.array(X_data)
y_data = np.array(y_data)

split_len = int(len(X_data)*0.7)

X_train = X_data[0:split_len]
y_train = y_data[0:split_len]

X_test = X_data[split_len:]
y_test = y_data[split_len:]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = None
try:
    model = load_model("./Stock_prediction_Model_2.keras")
except Exception as e:
    print("Model file not found")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=64, epochs=epochs)

predictions = model.predict(y_test)
print("Printing predictions....")
print(predictions)
inv_pred = scaler.inverse_transform(predictions)
inv_y = scaler.inverse_transform(y_test)

res = math.sqrt(mean_squared_error(inv_y, inv_pred))
print(res)


data_plot = pd.DataFrame({
'original_test_data': inv_y.reshape(-1),
'predicted_test_data': inv_pred.reshape(-1)
}, index=stock_data.index[split_len + past_mem:])

print(data_plot.head())

st.pyplot(plot_graph((15, 5), data_plot, "Predictions"))

st.pyplot(plot_graph((15, 5), pd.concat([ stock_data_att[:split_len+past_mem], data_plot], axis=0), "Total"))