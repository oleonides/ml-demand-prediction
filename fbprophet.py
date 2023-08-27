import streamlit as st
import pandas as pd
import numpy as np

from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y']))
    fig.layout.update(
        title_text='Time series data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


def plot_test_data(test_data: pd.DataFrame, forecast: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_data['ds'], y=test_data['y'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.update_layout(title='Actual vs. Forecast',
                      xaxis_title='Date', yaxis_title='Qty')
    st.plotly_chart(fig)


def display_metrics(test_data: pd.DataFrame, forecast: pd.DataFrame):
    st.subheader('Metrics')
    r2 = r2_score(test_data['y'], forecast['yhat'])
    st.write(f'**Squared R:** {r2}')

    mae = mean_absolute_error(test_data['y'], forecast['yhat'])
    st.write(f'**Mean Absolute Error:** {mae}')

    mape = (mae / np.mean(test_data['y'])) * 100
    st.write(f'**Mean Absolute Percentaje Error:** {mape}%')


def forecast(data: pd.DataFrame, years: int):
    period = years * 365
    train_data, test_data = train_test_split(
        data, test_size=0.2, shuffle=False)

    m = Prophet()
    m.fit(train_data)

    future = m.make_future_dataframe(periods=len(test_data))
    forecast = m.predict(future)

    display_metrics(test_data, forecast.loc[len(train_data):])

    st.markdown("---")

    plot_test_data(test_data, forecast.loc[len(train_data):])

    st.markdown("---")

    start = st.checkbox("Start forecast")
    if start:
        m = Prophet()
        m.fit(data)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        st.subheader('Forecast data')
        st.write(forecast.tail())

        st.write(f'### Forecast plot for {years} years')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write("### Forecast components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)


def predict_demand_prophet(data: pd.DataFrame):
    st.write("### Demand Forecasting")
    n_years = st.slider('Years of prediction:', 1, 4)

    columns = data.columns.sort_values().tolist()
    ds = st.selectbox("Select DS variable", [None] + columns)
    y = st.selectbox("Select y variable", [None] + columns)

    if ds is not None and y is not None and ds != y:
        new_data = data[[ds, y]]

        y_options = [None, "Remove negative values", "Take absolute values"]
        y_selected_option = st.radio("Select an option", y_options)
        if y_selected_option is not None and y_selected_option == "Remove negative values":
            new_data = new_data[new_data[y] >= 0]
        if y_selected_option is not None and y_selected_option == "Take absolute values":
            new_data[y] = abs(new_data[y])

        group_values = st.checkbox("Group values by date")
        if group_values:
            new_data[ds] = pd.to_datetime(new_data[ds]).dt.strftime('%Y-%m-%d')
            new_data = new_data.groupby(ds)[y].sum().reset_index()

        new_data = new_data.rename(columns={ds: "ds", y: "y"})
        st.write(new_data)
        plot_raw_data(new_data)

        start = st.checkbox("Train model")
        if start:
            forecast(new_data, n_years)
