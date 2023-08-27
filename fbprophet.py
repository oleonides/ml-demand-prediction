import streamlit as st
import pandas as pd

from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly


def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y']))
    fig.layout.update(
        title_text='Time series data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


def predict_demand_prophet(data: pd.DataFrame):
    st.write("### Demand Forecasting")
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    columns = data.columns.sort_values().tolist()
    ds = st.selectbox("Select DS variable", [None] + columns)
    y = st.selectbox("Select y variable", [None] + columns)

    if ds is not None and y is not None and ds != y:
        df_train = data[[ds, y]]

        y_options = [None, "Remove negative values", "Take absolute values"]
        y_selected_option = st.radio("Select an option", y_options)
        if y_selected_option is not None and y_selected_option == "Remove negative values":
            df_train = df_train[df_train[y] >= 0]
        if y_selected_option is not None and y_selected_option == "Take absolute values":
            df_train[y] = abs(df_train[y])

        group_values = st.checkbox("Group values by date")
        if group_values:
            df_train[ds] = pd.to_datetime(df_train[ds]).dt.strftime('%Y-%m-%d')
            df_train = df_train.groupby(ds)[y].sum().reset_index()

        df_train = df_train.rename(columns={ds: "ds", y: "y"})
        st.write(df_train)
        plot_raw_data(df_train)

        start = st.checkbox("Start forecasting")
        if start:
            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)

            st.subheader('Forecast data')
            st.write(forecast.tail())

            st.write(f'### Forecast plot for {n_years} years')
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)

            st.write("### Forecast components")
            fig2 = m.plot_components(forecast)
            st.write(fig2)
