from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
import logging
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the data
file_path = '/analysis2/climatechangedata.xlsx'
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

try:
    data = pd.read_excel(file_path)
    data['Year'] = pd.to_datetime(data['Year'], format='%Y')
    data = data.set_index('Year')
except Exception as e:
    logging.error(f"Error loading or parsing data: {e}")
    raise

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

@app.route('/')
def index():
    return render_template('index.html', variables=variables + ['Predict All Indicators'])

@app.route('/result', methods=['POST'])
def result():
    try:
        indicator = request.form['indicator']
        year = int(request.form['year'])

        logging.debug(f"Selected indicator: {indicator}")
        logging.debug(f"Selected year: {year}")

        if indicator not in data.columns and indicator != 'Predict All Indicators':
            error_message = f"Selected indicator '{indicator}' not found in the available indicators."
            return render_template('result.html', error=error_message)

        results = {}
        plot_data = []
        indicators_to_predict = variables if indicator == 'Predict All Indicators' else [indicator]

        for ind in indicators_to_predict:
            variable = data[ind]
            train_data = data[(data.index.year <= 2015) & (data.index.year >= 1990)][ind]
            test_data = data[(data.index.year >= 2016) & (data.index.year <= 2023)][ind]

            if len(train_data) == 0:
                results[ind] = "Insufficient data"
                continue

            try:
                stepwise_model = pm.auto_arima(train_data, start_p=1, start_q=1,
                                               max_p=5, max_q=5, m=12,
                                               seasonal=True, D=1,
                                               d=None, trace=False,
                                               error_action='ignore',
                                               suppress_warnings=True,
                                               stepwise=True)
            except Exception as e:
                error_message = f"An error occurred while modeling the data: {str(e)}"
                logging.error(error_message)
                results[ind] = error_message
                continue

            forecast_steps = year - 2016 + 1
            forecast = stepwise_model.predict(n_periods=forecast_steps)
            result = forecast[-1]

            # Formatting result based on indicator
            if ind in ['Carbon Dioxide (Million metric tons of CO2 equivalent)',
                       'Methane (Million metric tons of CO2 equivalent)',
                       'Nitrous Oxide (Million metric tons of CO2 equivalent)',
                       'Fluorinated Gases (Million metric tons of CO2 equivalent)',
                       'Total GHG (Million metric tons of CO2 equivalent)']:
                result = f"{result:.3f}"
            elif ind == 'Temperature (Celcius)':
                result = f"{result:.1f}°C"
                ind = "Average Temperature"
            elif ind == 'Forest Area (%)':
                result = f"%{result:.1f}"

            results[ind] = result

            # Prepare data for plotly
            plot_data.append({
                'x': list(train_data.index.year) + [y for y in range(2016, year + 1)],
                'y': list(train_data) + list(forecast),
                'type': 'scatter',
                'name': ind
            })


        return render_template('result.html', year=year, results=results, plot_data=plot_data)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return render_template('result.html', error="An unexpected error occurred. Please try again.")

if __name__ == '__main__':
    app.run(debug=True)
