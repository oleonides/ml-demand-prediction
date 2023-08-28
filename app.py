import streamlit as st
import pandas as pd
from sidebar import get_data
from fbprophet import predict_demand_prophet
from xgboost_forecasting import predict_demand_xgboost

data = get_data()


def select_and_rename_column(data: pd.DataFrame):
    st.info("If you want to rename a column, select the column and enter the new name.")
    all_columns = data.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to rename", options=all_columns)

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


def select_columns(data: pd.DataFrame):
    st.info("Select ✅ the columns to be used by the forecasting model")
    all_columns = data.columns.sort_values().tolist()
    selected_columns = st.multiselect("Select columns", options=all_columns)

    if selected_columns:
        sub_df = data[selected_columns]
        renamed_df = select_and_rename_column(sub_df)
        st.write("### New dataset")
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

    return select_columns(data)


if data is None:
    st.info("Upload a dataset or enter the URL of an online dataset to get started.")
else:
    tab1, tab2 = st.tabs(["Data analysis", "Forecasting"])
    with tab1:
        df = show_data_analysis(data)

    with tab2:
        if df is None:
            st.info("Select the columns to be used by the forecasting model")

        if df is not None and len(df.columns) >= 2:
            model = st.selectbox("Select ML model", ["Prophet", "XGBoost"])
            if model == "Prophet":
                predict_demand_prophet(df)
            else:
                predict_demand_xgboost(df)
