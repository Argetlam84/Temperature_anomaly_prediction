import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
from datetime import datetime

filtered_date = "2000"

def plot_filtered_forecast(time_series, fitted_values, forecast, title, filtered_date):
    # Convert filtered_date to a datetime object
    filtered_date = pd.to_datetime(filtered_date, format="%Y")

    # Slice the Series, not the index
    time_series_filtered = time_series.loc[filtered_date:]
    fitted_values_filtered = fitted_values.loc[filtered_date:]

    plt.plot(time_series_filtered, label='Original')
    plt.plot(fitted_values_filtered, label='Fitted', color='red')
    plt.plot(forecast.index, forecast, label='Forecast', color='green') 

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Anomaly')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

# Read the CSV file, skipping the first four lines
df = pd.read_csv("datasets/data.csv", skiprows=4, on_bad_lines='skip')
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m')
df_ = df.copy()
df.set_index("Date", inplace=True)
time_series = df["Anomaly"]

df_.rename(columns={"Date":"ds","Anomaly":"y"}, inplace=True)

# Streamlit app
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a Page", ["About", "Prediction"])

if page == "About":
    st.title("About")
    image_url ="https://github.com/user-attachments/assets/200257fb-04b2-4a16-940f-1743812afad3"
    st.image(image_url, use_column_width=True)
    st.write("""
    ### Time Series Forecasting App
    This application allows you to explore various time series forecasting models 
    and visualize the predictions. The following models are available:
    - Simple Exponential Smoothing (SES)
    - Holt’s Linear Trend Model
    - Holt-Winters Seasonal Model
    - ARIMA
    - SARIMA
    - SARIMA
    - Prophet         

    Use the navigation sidebar to switch between the main page and the prediction page. 
    On the prediction page, you can select a model and specify the number of steps to forecast.
    The results will be displayed as plots showing the original data, the fitted values, 
    and the forecasted values.
    """)

elif page == "Prediction":
    st.title("Prediction Page")
    
    model_type = st.selectbox("Select Model Type", ["SES", "Holt", "Holt-Winters", "ARIMA", "SARIMA", "Prophet"])
    
    user_input = st.number_input("Input a value for forecasting:", min_value=1, max_value=100, step=1)
    
    if 'forecast_button' not in st.session_state:
        st.session_state.forecast_button = False

    def forecast():
        st.session_state.forecast_button = True

    def run_forecast(model_type):
        if model_type == "SES":
            model = joblib.load("ses_model.pkl")
            forecast = model.forecast(steps=user_input)
            plot_filtered_forecast(time_series, model.fittedvalues, forecast, "SES", filtered_date)

        elif model_type == "Holt":
            model = joblib.load("holt_fit.pkl")
            forecast = model.forecast(steps=user_input)
            plot_filtered_forecast(time_series, model.fittedvalues, forecast, "Holt", filtered_date)
        
        elif model_type == "Holt-Winters":
            model = joblib.load("holt_winters_fit.pkl")
            forecast = model.forecast(steps=user_input)
            plot_filtered_forecast(time_series, model.fittedvalues, forecast, "Holt-Winters", filtered_date)
        
        elif model_type == "ARIMA":
            model = joblib.load("arima_fit.pkl")
            forecast = model.forecast(steps=user_input)
            plot_filtered_forecast(time_series, model.fittedvalues, forecast, "ARIMA", filtered_date)
        
        elif model_type == "SARIMA":
            model = joblib.load("sarima_fit.pkl")
            forecast = model.forecast(steps=user_input)
            plot_filtered_forecast(time_series, model.fittedvalues, forecast, "SARIMA", filtered_date)

        elif model_type == "Prophet":
            model = joblib.load("prophet_model.pkl")
            future = model.make_future_dataframe(periods=user_input, freq="M")
            forecast = model.predict(future)

            forecast["ds"] = pd.to_datetime(forecast["ds"])
            df_["ds"] = pd.to_datetime(df_["ds"])

            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_["ds"], df_['y'], label='Real Anomalies', color='blue')
            ax.plot(forecast["ds"], forecast["yhat"], label='Predicted Anomalies', color='red')
            ax.set_title('Anomaly Forecasting with Prophet')
            ax.set_xlabel('Date')
            ax.set_ylabel('Anomaly')
            ax.legend()
            st.pyplot(fig)

    if st.button(f"Run {model_type} Model", on_click=forecast):
        if st.session_state.forecast_button:
            run_forecast(model_type)
