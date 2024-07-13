# # from flask import Flask, request, jsonify, render_template
# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import MinMaxScaler
# # from keras.models import load_model
# # import yfinance as yf
# # import datetime
# # import os

# # # Disable oneDNN custom operations
# # os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# # import tensorflow as tf

# # app = Flask(__name__)

# # # Load the pre-trained model
# # model = load_model('forecast.h5')
# # scaler = MinMaxScaler(feature_range=(0, 1))

# # def predict_stock(symbol):
# #     start = '2015-01-01'
# #     end = datetime.datetime.today().strftime('%Y-%m-%d')
    
# #     # Fetch data
# #     df = yf.download(symbol, start=start, end=end)
    
# #     if df.empty:
# #         return {"error": "Invalid stock symbol or no data available"}
    
# #     # Prepare data
# #     data = df.filter(['Close'])
# #     dataset = data.values
# #     training_data_len = int(np.ceil(len(dataset) * 0.8))
    
# #     scaled_data = scaler.fit_transform(dataset)
    
# #     train_data = scaled_data[0:training_data_len, :]
# #     x_train = []
# #     y_train = []
    
# #     for i in range(90, len(train_data)):
# #         x_train.append(train_data[i-90:i, 0])
# #         y_train.append(train_data[i, 0])
    
# #     x_train, y_train = np.array(x_train), np.array(y_train)
# #     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
# #     test_data = scaled_data[training_data_len - 90:, :]
# #     x_test = []
# #     y_test = dataset[training_data_len:, :]
    
# #     for i in range(90, len(test_data)):
# #         x_test.append(test_data[i-90:i, 0])
    
# #     x_test = np.array(x_test)
# #     x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
# #     predictions = model.predict(x_test)
# #     predictions = scaler.inverse_transform(predictions)
    
# #     # Get the dates for the predictions
# #     last_date = df.index[-1]
# #     dates = [last_date + datetime.timedelta(days=i) for i in range(1, len(predictions) + 1)]
    
# #     # Prepare the response
# #     forecast = {str(dates[i]): float(predictions[i]) for i in range(len(predictions))}
    
# #     return forecast

# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     data = request.get_json()
# #     symbol = data.get('symbol')
    
# #     if not symbol:
# #         return jsonify({"error": "No stock symbol provided"}), 400
    
# #     forecast = predict_stock(symbol)
    
# #     return jsonify(forecast)

# # if __name__ == '__main__':
# #     app.run(debug=True)


# from flask import Flask, request, jsonify, render_template, send_from_directory
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import load_model
# import yfinance as yf
# import datetime
# import os
# import matplotlib.pyplot as plt

# # Disable oneDNN custom operations
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import tensorflow as tf

# app = Flask(__name__)

# # Load the pre-trained model
# model = load_model('forecast.h5')
# scaler = MinMaxScaler(feature_range=(0, 1))

# # Generate and save the plot
# def create_plot(train, valid):
#     plt.figure(figsize=(16,8))
#     plt.title('Model')
#     plt.xlabel('Data', fontsize=18)
#     plt.ylabel('Close Price USD ($)', fontsize=18)
#     plt.plot(train['Close'])
#     plt.plot(valid[['Close', 'Predictions']])
#     plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
#     plt.savefig('static/plot.png')
#     plt.close()

# def predict_stock(symbol):
#     start = '2015-01-01'
#     end = datetime.datetime.today().strftime('%Y-%m-%d')
    
#     # Fetch data
#     df = yf.download(symbol, start=start, end=end)
    
#     if df.empty:
#         return {"error": "Invalid stock symbol or no data available"}
    
#     # Prepare data
#     data = df.filter(['Close'])
#     dataset = data.values
#     training_data_len = int(np.ceil(len(dataset) * 0.8))
    
#     scaled_data = scaler.fit_transform(dataset)
    
#     train_data = scaled_data[0:training_data_len, :]
#     x_train = []
#     y_train = []
    
#     for i in range(90, len(train_data)):
#         x_train.append(train_data[i-90:i, 0])
#         y_train.append(train_data[i, 0])
    
#     x_train, y_train = np.array(x_train), np.array(y_train)
#     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
#     test_data = scaled_data[training_data_len - 90:, :]
#     x_test = []
#     y_test = dataset[training_data_len:, :]
    
#     for i in range(90, len(test_data)):
#         x_test.append(test_data[i-90:i, 0])
    
#     x_test = np.array(x_test)
#     x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
#     predictions = model.predict(x_test)
#     predictions = scaler.inverse_transform(predictions)
    
#     # Generate plot
#     train = data[:training_data_len]
#     valid = data[training_data_len:]
#     valid['Predictions'] = predictions
#     create_plot(train, valid)
    
#     # Get the dates for the predictions
#     last_date = df.index[-1]
#     dates = [last_date + datetime.timedelta(days=i) for i in range(1, len(predictions) + 1)]
    
#     # Prepare the response
#     forecast = {str(dates[i]): float(predictions[i]) for i in range(len(predictions))}
    
#     return forecast

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/plot.png')
# def plot_png():
#     return send_from_directory('static', 'plot.png')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     symbol = data.get('symbol')
    
#     if not symbol:
#         return jsonify({"error": "No stock symbol provided"}), 400
    
#     forecast = predict_stock(symbol)
    
#     return jsonify(forecast)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import yfinance as yf
import datetime
import os
import matplotlib.pyplot as plt
import uuid

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model
model = load_model('forecast.h5')
scaler = MinMaxScaler(feature_range=(0, 1))

# Generate and save the plot
def create_plot(train, valid, plot_filename):
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.savefig(f'static/{plot_filename}')
    plt.close()

def predict_stock(symbol):
    start = '2015-01-01'
    end = datetime.datetime.today().strftime('%Y-%m-%d')
    
    # Fetch data
    df = yf.download(symbol, start=start, end=end)
    
    if df.empty:
        return {"error": "Invalid stock symbol or no data available"}
    
    # Prepare data
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * 0.8))
    
    scaled_data = scaler.fit_transform(dataset)
    
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []
    
    for i in range(90, len(train_data)):
        x_train.append(train_data[i-90:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    test_data = scaled_data[training_data_len - 90:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    
    for i in range(90, len(test_data)):
        x_test.append(test_data[i-90:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Generate plot
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    
    # Create a unique filename for the plot
    plot_filename = f'plot_{uuid.uuid4().hex}.png'
    create_plot(train, valid, plot_filename)
    
    # Get the dates for the predictions
    last_date = df.index[-1]
    dates = [last_date + datetime.timedelta(days=i) for i in range(1, len(predictions) + 1)]
    
    # Prepare the response
    forecast = {str(dates[i]): float(predictions[i]) for i in range(len(predictions))}
    
    return forecast, plot_filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symbol = data.get('symbol')
    
    if not symbol:
        return jsonify({"error": "No stock symbol provided"}), 400
    
    forecast, plot_filename = predict_stock(symbol)
    
    return jsonify({"forecast": forecast, "plot_filename": plot_filename})

if __name__ == '__main__':
    app.run(debug=True)
