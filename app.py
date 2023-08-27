import streamlit as st
import pandas as pd
import plotly.express as px
from sidebar import get_data

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

data = get_data()


def select_and_rename_column(data: pd.DataFrame):
    st.write("### Select and Rename Columns")

    # Select columns to rename
    all_columns = data.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to rename", options=all_columns)

    # Rename the selected columns
    for column in selected_columns:
        new_column_name = st.text_input(
            f"Enter new name for column '{column}'", value=column)
        if column != new_column_name:
            data.rename(columns={column: new_column_name}, inplace=True)
            st.write(
                f"Column '{column}' renamed as '{new_column_name}' successfully!")

    return data


def show_data_correlation(data):
    st.write("Data Correlation")
    st.write(data.corr(numeric_only=True))


def show_missing_values(data):
    st.write("Missing Values")
    st.write(data.isnull().sum())


def show_percent_missing(data):
    st.write("Missing Percentage")
    st.write(data.isna().mean().mul(100))


def show_unique_values(data):
    st.write("Unique Values")
    st.write(data.nunique())


def show_data_shape(data):
    st.write("Number of rows")
    st.write(data.shape[0])
    st.write("Number of columns")
    st.write(data.shape[1])

#################### Forecasting ####################
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['DATE_ONLY'],
                  y=data['QUANTITY'], name="stock_open"))
    fig.layout.update(
        title_text='Time series data with range slider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


def predict_demand_prophet(data):
    st.write("### Demand Forecasting")
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    data['QUANTITY'] = abs(data['QUANTITY'])
    data['CREATED_DATE_UTC'] = pd.to_datetime(data['CREATED_DATE_UTC'])
    data['DATE_ONLY'] = pd.to_datetime(data['CREATED_DATE_UTC'].dt.strftime('%Y-%m-%d'))

    df_train = data[['DATE_ONLY', 'QUANTITY']]
    final_df = df_train.groupby('DATE_ONLY')['QUANTITY'].sum().reset_index()
    plot_raw_data(final_df)

    # Predict forecast with Prophet.
    final_df = df_train.rename(columns={"DATE_ONLY": "ds", "QUANTITY": "y"})

    m = Prophet()
    m.fit(final_df)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)


def select_columns(data: pd.DataFrame):
    st.write("### Select Columns")
    all_columns = data.columns.tolist()
    selected_columns = st.multiselect("Select columns", options=all_columns)

    if selected_columns:
        sub_df = data[selected_columns]
        renamed_df = select_and_rename_column(sub_df)
        st.write("### Sub DataFrame")
        st.write(renamed_df.head())

        st.write("Description")
        st.write(renamed_df.describe().T)

        show_data_shape(renamed_df)

        col1, col2, col3 = st.columns(3)
        with col1:
            show_missing_values(renamed_df)
        with col2:
            show_percent_missing(renamed_df)
        with col3:
            show_unique_values(renamed_df)

        show_data_correlation(renamed_df)

        predict_demand_prophet(renamed_df)
    else:
        st.warning("Please select at least one column.")


if data is not None:
    st.write('Data Dimension: ' +
             str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')
    st.write(data)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Columns ", data.columns)
    with col2:
        st.write("Data Types ", data.dtypes)

    all_columns = data.columns.tolist()
    select_columns(data)






