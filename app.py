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
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y']))
    fig.layout.update(
        title_text='Time series data with range slider', xaxis_rangeslider_visible=True)
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

            st.write(f'Forecast plot for {n_years} years')
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)

            st.write("Forecast components")
            fig2 = m.plot_components(forecast)
            st.write(fig2)


def predict_demand_xgboost(data):
    return data


def select_columns(data: pd.DataFrame):
    st.write("### Select Columns")
    all_columns = data.columns.sort_values().tolist()
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
        return renamed_df

    st.info("Please select at least one column.")
    return None


def show_data_analysis(data: pd.DataFrame):
    st.write('Data Dimension: ' +
             str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')
    st.write(data)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Columns ", data.columns)
    with col2:
        st.write("Data Types ", data.dtypes)

    all_columns = data.columns.tolist()
    return select_columns(data)


if data is None:
    st.info("Upload a dataset or enter the URL of an online dataset to get started.")
else:
    tab1, tab2 = st.tabs(["Data analysis", "Forecasting"])
    with tab1:
        df = show_data_analysis(data)

    with tab2:
        if df is not None and len(df.columns) >= 2:
            model = st.selectbox("Select ML model", ["XGBoost", "FB Prophet"])
            if model == "FB Prophet":
                predict_demand_prophet(df)
            else:
                predict_demand_xgboost(df)
