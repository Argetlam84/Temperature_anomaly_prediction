# Temperature Anomaly Prediction

This repository contains a Jupyter Notebook for analyzing and forecasting global temperature anomalies using various time series models. The notebook demonstrates step-by-step data processing, exploration, and prediction techniques suitable for both beginners and those with some experience in data science and time series analysis.

<a href="https://temperatureanomalyprediction-argetlam.streamlit.app/" target="_blank">Overview </a>

The main objective of this project is to predict temperature anomalies over time, helping to understand trends and potential future changes in global temperatures. We use historical data and apply several forecasting models to generate predictions.

## Features

- **Data Loading and Preparation**: Reads a CSV file containing temperature anomaly data, processes the data, and prepares it for analysis.
- **Data Exploration**: Provides initial insights into the dataset, including viewing the first few records to understand its structure.
- **Time Series Forecasting Models**:
  - **Simple Exponential Smoothing**: A statistical technique used to forecast data with no trend or seasonal patterns.
  - **Holt-Winters Exponential Smoothing**: Extends simple exponential smoothing to capture trends and seasonality.
  - **ARIMA (AutoRegressive Integrated Moving Average)**: A popular model for time series forecasting that can capture different types of patterns.
  - **SARIMAX (Seasonal AutoRegressive Integrated Moving-Average with eXogenous regressors)**: An extension of ARIMA that supports seasonality and external variables.
  - **Prophet**: A forecasting tool developed by Facebook that works well with time series data with daily observations that display seasonal effects.

## Getting Started

To get a local copy of the project and run the notebook, follow these steps:

### Prerequisites

Ensure you have the following installed on your system:

- Python (>=3.7)
- Jupyter Notebook or JupyterLab
- Required Python libraries: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `prophet`, `joblib`

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib statsmodels prophet joblib

