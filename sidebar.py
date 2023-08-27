import streamlit as st
import pandas as pd
import requests

st.sidebar.subheader('Query parameters')


@st.cache_resource
def upload_file(file):
    """
    Load data from a file (CSV or Excel).

    Parameters:
        file (File): The file to load.

    Returns:
        DataFrame: The loaded data.
    """
    file_extension = file.name.split(".")[-1]
    if file_extension == "csv":
        data = pd.read_csv(file)
    elif file_extension in ["xls", "xlsx"]:
        data = pd.read_excel(file)
    else:
        st.warning("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    return data


def get_data():
    file_option = st.sidebar.radio("Data Source",
                                   options=["Upload Local File", "Enter Online Dataset"])
    file = None
    data = None

    if file_option == "Upload Local File":
        file = st.sidebar.file_uploader(
            "Upload a dataset in CSV or EXCEL format", type=["csv", "excel"])

    elif file_option == "Enter Online Dataset":
        online_dataset = st.sidebar.text_input(
            "Enter the URL of the online dataset")
        if online_dataset:
            try:
                response = requests.get(online_dataset, timeout=5)
                if response.ok:
                    data = pd.read_csv(online_dataset)
                else:
                    st.warning(
                        "Unable to fetch the dataset from the provided link.")
            except:
                st.warning(
                    "Invalid URL or unable to read the dataset from the provided link.")

    if file is not None:
        data = upload_file(file)

    return data

