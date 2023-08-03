import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import warnings
from scipy.stats import lognorm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
import joblib
import os

warnings.filterwarnings('ignore')

# Assuming you have already loaded the CSV data into the 'data' DataFrame
data = pd.read_csv('NIFTY50.csv')

# Convert the 'datetime' column to datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# Filter data for the years 2017 to 2019
start_date = pd.to_datetime('2017-01-01')
end_date = pd.to_datetime('2019-12-31')
filtered_data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]

# Select only the 'close' column
data = filtered_data[['close']]

# Fit a lognormal distribution to the 'close' data
mu, sigma = np.log(data['close']).mean(), np.log(data['close']).std()
s = np.random.lognormal(mu, sigma, len(data))

scaler = StandardScaler()
data_for_scaling = s
data_for_scaling = data_for_scaling.reshape(-1, 1)

data_scaled = scaler.fit_transform(data_for_scaling)

data_scaled_df = pd.DataFrame(data_scaled, columns=['log close'])

# Check if the trained model file exists, if not, fit the KMeans model and save it
if not os.path.exists('kmeans_model.joblib'):
    model = KMeans(n_clusters=3, init='k-means++')
    model.fit(data_scaled_df)
    # Save the trained model to a file
    joblib.dump(model, 'kmeans_model.joblib')
else:
    # Load the trained model from the file
    model = joblib.load('kmeans_model.joblib')


# Function to predict cluster
def predict_cluster(closing_price):
    scaled_data = scaler.transform([[closing_price]])
    prediction = model.predict(scaled_data)
    if prediction[0] == 0:
        regime = 'Bearish'
    elif prediction[0] == 2:
        regime = 'Consolidated'
    else:
        regime = 'Bullish'
    return regime

# Take input
closing_price = st.number_input('Enter closing price: ')

# Predict cluster
if closing_price is not None:
    # Predict cluster
    cluster = predict_cluster(closing_price)
    # Print result
    st.text(f"Closing price {closing_price} belongs to cluster: {cluster}")
