import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

from plotly import graph_objs as go
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


def plot_raw_data(data, outcome: str):
    grouped_data = data[['year', 'month', outcome]]
    st.write(grouped_data)
    grouped_data = grouped_data.groupby(['year', 'month'])[
        outcome].sum().reset_index()
    fig = px.line(grouped_data, x='month', y=outcome,
                  color='year', markers=True)
    fig.update_layout(title='Monthly Quantity Over Years',
                      xaxis_title='Month',
                      yaxis_title='Quantity')
    st.plotly_chart(fig)


def plot_test_data(y_test: pd.DataFrame, xgb_fcst: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(len(y_test))),
                  y=y_test, mode='lines', name='y_test'))
    fig.add_trace(go.Scatter(x=list(range(len(xgb_fcst))),
                  y=xgb_fcst, mode='lines', name='xgb_fcst'))

    fig.update_layout(title='Comparison between y_test and xgb_fcst',
                      xaxis_title='Steps in Test Data',
                      yaxis_title='Values')

    st.plotly_chart(fig)


def display_metrics(test_data: pd.DataFrame, forecast: pd.DataFrame):
    st.subheader('Metrics')
    r2 = r2_score(test_data, forecast)
    st.write(f'**Squared R:** {r2}')

    mae = mean_absolute_error(test_data, forecast)
    st.write(f'**Mean Absolute Error:** {mae}')

    mape = (mae / np.mean(test_data)) * 100
    st.write(f'**Mean Absolute Percentaje Error:** {mape}%')


def train_model(X: pd.DataFrame, y: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=12345, shuffle=True)
    my_xgb = XGBRegressor()
    my_xgb.fit(X_train, y_train)
    xgb_fcst = my_xgb.predict(X_test)
    display_metrics(y_test, xgb_fcst)
    plot_test_data(y_test, xgb_fcst)


def predict_demand_xgboost(data: pd.DataFrame):
    columns = data.columns.sort_values().tolist()
    predictor_variables = st.multiselect("Select predictor variables", columns)
    y = st.selectbox("Select outcome variable", [None] + columns)

    if predictor_variables is not None and y is not None and y not in predictor_variables:
        new_data = data[predictor_variables + [y]]
        y_options = [None, "Remove negative values", "Take absolute values"]
        y_selected_option = st.radio("Select an option", y_options)
        if y_selected_option is not None and y_selected_option == "Remove negative values":
            new_data = new_data[new_data[y] >= 0]
        if y_selected_option is not None and y_selected_option == "Take absolute values":
            new_data[y] = abs(new_data[y])

        split_date = st.checkbox("split date into year, month, day")
        if split_date:
            date_column = st.selectbox("Select date column", [
                                       None] + predictor_variables)
            if date_column is not None:
                new_data['year'] = new_data[date_column].apply(
                    lambda x: int(x[:4]))
                new_data['month'] = new_data[date_column].apply(
                    lambda x: int(x[5:7]))
                new_data['weekday'] = new_data[date_column].apply(
                    lambda x: int(x[8:10]))

                new_data.drop(date_column, axis=1, inplace=True)
                predictor_variables.remove(date_column)
                predictor_variables = predictor_variables + \
                    ['year', 'month', 'weekday']

        group_values = st.checkbox("Group values by predictor variables")
        if group_values:
            new_data = new_data.groupby(predictor_variables)[
                y].sum().reset_index()
            plot_raw_data(new_data, y)

        start_training = st.checkbox("Train model")
        if start_training:
            X = new_data[predictor_variables]
            y = new_data[y]
            train_model(X, y)
