import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from io import StringIO

# Custom function to load data with error handling for bad lines
def custom_load_data():
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
    if uploaded_file is not None:
        delimiter = st.text_input("Enter the delimiter (default is ',')", value=',')
        header = st.text_input("Enter the header row number (default is 0)", value=0)
        header = int(header)
        bad_lines = []

        try:
            # Load CSV with custom error handling
            string_io = StringIO(uploaded_file.getvalue().decode("utf-8"))
            reader = csv.reader(string_io, delimiter=delimiter)
            headers = next(reader)
            rows = []

            for row in reader:
                if len(row) == len(headers):
                    rows.append(row)
                else:
                    bad_lines.append(row)

            df = pd.DataFrame(rows, columns=headers)

            if bad_lines:
                st.warning(f"Skipped {len(bad_lines)} bad lines.")
                if st.checkbox("Show bad lines"):
                    st.write(bad_lines[:5])  # Show first 5 bad lines

            if df.empty:
                st.error("The resulting dataframe is empty. Please check the delimiter and header settings.")
                return None

            return df
        except Exception as e:
            st.error(f"Error parsing file: {e}")

    return None

# Function to handle missing values
def handle_missing_values(data, method):
    if method == "Delete rows":
        return data.dropna()
    elif method == "Delete columns":
        return data.dropna(axis=1)
    elif method == "Replace with mean":
        imputer = SimpleImputer(strategy='mean')
    elif method == "Replace with median":
        imputer = SimpleImputer(strategy='median')
    elif method == "Replace with mode":
        imputer = SimpleImputer(strategy='most_frequent')
    elif method == "KNN imputation":
        imputer = KNNImputer()
    else:
        return data
    return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Function to normalize data
def normalize_data(data, method):
    if method == "Min-Max":
        scaler = MinMaxScaler()
    elif method == "Z-score":
        scaler = StandardScaler()
    else:
        return data
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Function to plot histograms
def plot_histograms(data):
    for column in data.select_dtypes(include=[np.number]).columns:
        fig, ax = plt.subplots()
        sns.histplot(data[column], kde=True, ax=ax)
        st.pyplot(fig)

# Function to plot box plots
def plot_box_plots(data):
    for column in data.select_dtypes(include=[np.number]).columns:
        fig, ax = plt.subplots()
        sns.boxplot(data[column], ax=ax)
        st.pyplot(fig)

# Function to apply clustering
def apply_clustering(data, algorithm, params):
    if algorithm == "K-Means":
        model = KMeans(n_clusters=params['n_clusters'])
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    else:
        return data
    clusters = model.fit_predict(data)
    data['Cluster'] = clusters
    return data, model

# Streamlit app with sidebar menu
with st.sidebar:
    selected = st.selectbox(
        "Main Menu",
        ["Home", "Data Exploration", "Data Preprocessing", "Visualization", "Clustering"]
    )

if selected == "Home":
    st.title("Interactive Data Analysis and Clustering Application")
    st.header("Welcome to the Home Page")

elif selected == "Data Exploration":
    st.title("Data Exploration")
    data = custom_load_data()
    if data is not None:
        st.subheader("Data Preview")
        st.write(data.head())
        st.write(data.tail())
        st.subheader("Data Description")
        st.write(data.describe())
        st.write("Number of rows:", data.shape[0])
        st.write("Number of columns:", data.shape[1])
        st.write("Column names:", data.columns.tolist())
        st.write("Missing values per column:", data.isnull().sum())

elif selected == "Data Preprocessing":
    st.title("Data Pre-processing and Cleaning")
    data = custom_load_data()
    if data is not None:
        missing_values_option = st.selectbox("Choose a method to handle missing values",
                                             ["Delete rows", "Delete columns", "Replace with mean", "Replace with median", "Replace with mode", "KNN imputation"])
        data = handle_missing_values(data, missing_values_option)
        
        st.write("Data after handling missing values")
        st.write(data.head())

        normalization_option = st.selectbox("Choose a normalization method", ["None", "Min-Max", "Z-score"])
        data = normalize_data(data, normalization_option)

        st.write("Data after normalization")
        st.write(data.head())

elif selected == "Visualization":
    st.title("Data Visualization")
    data = custom_load_data()
    if data is not None:
        st.write("Histograms")
        plot_histograms(data)

        st.write("Box Plots")
        plot_box_plots(data)

elif selected == "Clustering":
    st.title("Clustering")
    data = custom_load_data()
    if data is not None:
        clustering_algorithm = st.selectbox("Choose a clustering algorithm", ["K-Means", "DBSCAN"])
        if clustering_algorithm == "K-Means":
            n_clusters = st.slider("Choose the number of clusters", 2, 10)
            params = {'n_clusters': n_clusters}
        elif clustering_algorithm == "DBSCAN":
            eps = st.slider("Choose the epsilon value", 0.1, 10.0)
            min_samples = st.slider("Choose the minimum number of samples", 1, 10)
            params = {'eps': eps, 'min_samples': min_samples}
        
        data, model = apply_clustering(data, clustering_algorithm, params)
        
        st.write("Clustered Data")
        st.write(data.head())

        st.subheader("Cluster Visualization")
        if 'Cluster' in data.columns:
            fig, ax = plt.subplots()
            sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue='Cluster', data=data, palette='viridis', ax=ax)
            st.pyplot(fig)

        st.subheader("Cluster Statistics")
        if clustering_algorithm == "K-Means":
            st.write("Cluster Centers")
            st.write(model.cluster_centers_)
        st.write("Number of data points in each cluster")
        st.write(data['Cluster'].value_counts())

else:
    st.write("Please select an option from the menu to get started")
