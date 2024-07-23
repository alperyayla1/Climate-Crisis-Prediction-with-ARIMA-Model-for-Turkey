import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

import pmdarima as pm




def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Load the data
data = pd.read_excel('C:/Users/alper/OneDrive/Masaüstü/Yeni klasör/analysis2\climatechangedata.xlsx')
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data = data.set_index('Year')

# Selecting the variables for analysis
variables = [
    'Carbon Dioxide (Million metric tons of CO2 equivalent)',
    'Methane (Million metric tons of CO2 equivalent)',
    'Nitrous Oxide (Million metric tons of CO2 equivalent)',
    'Fluorinated Gases (Million metric tons of CO2 equivalent)',
    'Total GHG (Million metric tons of CO2 equivalent)',
    'Temperature (Celcius)',
    'Forest Area (%)'
]

# Plotting historical data and forecasting for each variable
for variable in variables:
    print(f"Variable: {variable}")

    train_data = data[(data.index.year < 2016) & (data.index.year >= 1990)][variable]
    test_data = data[(data.index.year >= 2016) & (data.index.year <= 2023)][variable]

    # ARIMA Model
    stepwise_model = pm.auto_arima(train_data, start_p=1, start_q=1,
                                   max_p=5, max_q=5, m=12,
                                   seasonal=True, D=1,
                                   d=None, trace=False,
                                   error_action='ignore',
                                   suppress_warnings=True,
                                   stepwise=True)

    # ARIMA Forecasting
    forecast_steps = 2056 - 2016
    forecast = stepwise_model.predict(n_periods=forecast_steps)

    # ARIMA Metrics
    mae = mean_absolute_error(test_data, forecast[:len(test_data)])
    mape = mean_absolute_percentage_error(test_data, forecast[:len(test_data)])
    print(f"ARIMA - MAE: {mae:.4f}, MAPE: {mape:.4f}%")

    # Prepare data for LSTM





    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data, label='Historical Data', marker='o')
    plt.plot(test_data.index, test_data, label='Testing Data', marker='o')
    plt.plot(pd.date_range(start='2016', periods=len(forecast), freq='YS'), forecast, label='ARIMA Forecast',
             marker='o', linestyle='--', color = 'red')

    plt.title(f'Time Series Analysis of {variable}')
    plt.xlabel('Year')
    plt.ylabel(variable)
    plt.legend()
    plt.grid(True)
    plt.show()