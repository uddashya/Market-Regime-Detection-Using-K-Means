import pandas as pd
import numpy as np
import random as rd
import warnings
from scipy.stats import lognorm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
import joblib
import os

warnings.filterwarnings('ignore')

data = pd.read_csv('NIFTY50.csv')

# Convert the 'datetime' column to datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# Filter data for the years 2017 to 2019
start_date = pd.to_datetime('2017-01-01')
end_date = pd.to_datetime('2019-12-31')
filtered_data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]

# Select only the 'close' column
data = filtered_data[['close']]

# Set the 'datetime' column as the index
filtered_data.set_index('datetime', inplace=True)

# Make sure the index is in DatetimeIndex format
filtered_data.index = pd.DatetimeIndex(filtered_data.index)


# Resample the data on a weekly basis and calculate OHLCV values
monthly_resampled_data = filtered_data.resample('M').apply({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})


# Resample the data on a weekly basis and calculate OHLCV values
weekly_resampled_data = filtered_data.resample('W').apply({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})


# Fit a lognormal distribution to the 'close' data
mu_weekly, sigma_weekly = np.log(weekly_resampled_data['close']).mean(), np.log(weekly_resampled_data['close']).std()
s_weekly = np.random.lognormal(mu_weekly, sigma_weekly, len(weekly_resampled_data))
# Fit a lognormal distribution to the 'close' data
mu, sigma = np.log(data['close']).mean(), np.log(data['close']).std()
s = np.random.lognormal(mu, sigma, len(data))
# Fit a lognormal distribution to the 'close' data
mu_monthly, sigma_monthly = np.log(monthly_resampled_data['close']).mean(), np.log(monthly_resampled_data['close']).std()
s_monthly = np.random.lognormal(mu_monthly, sigma_monthly, len(monthly_resampled_data))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_for_scaling = s
data_for_scaling = data_for_scaling.reshape(-1, 1)

data_scaled = scaler.fit_transform(data_for_scaling)

data_scaled_df = pd.DataFrame(data_scaled, columns=['log close'])

data_for_scaling_weekly = s_weekly
data_for_scaling_weekly = data_for_scaling_weekly.reshape(-1, 1)

data_scaled_weekly = scaler.fit_transform(data_for_scaling_weekly)

data_scaled_weekly_df = pd.DataFrame(data_for_scaling_weekly, columns=['log close'])

data_for_scaling_monthly = s_monthly
data_for_scaling_monthly = data_for_scaling_monthly.reshape(-1, 1)

data_scaled_monthly = scaler.fit_transform(data_for_scaling_monthly)

data_scaled_monthly_df = pd.DataFrame(data_for_scaling_weekly, columns=['log close'])

# Check if the trained model file exists, if not, fit the KMeans model and save it
if not os.path.exists('kmeans_model.joblib'):
    model = KMeans(n_clusters=3, init='k-means++')
    model.fit(data_scaled_df)
    # Save the trained model to a file
    joblib.dump(model, 'kmeans_model.joblib')
else:
    # Load the trained model from the file
    model = joblib.load('kmeans_model.joblib')

data_scaled_df['Cluster'] = model.predict(data_scaled_df)

# Check if the trained model file exists for weekly data, if not, fit the KMeans model and save it
if not os.path.exists('kmeans_weekly_model.joblib'):
    model_weekly = KMeans(n_clusters=3, init='k-means++')
    model_weekly.fit(data_scaled_weekly_df)
    # Save the trained model to a file
    joblib.dump(model_weekly, 'kmeans_weekly_model.joblib')
else:
    # Load the trained model from the file
    model_weekly = joblib.load('kmeans_weekly_model.joblib')

data_scaled_weekly_df['Cluster'] = model_weekly.predict(data_scaled_weekly_df)

# Check if the trained model file exists for monthly data, if not, fit the KMeans model and save it
if not os.path.exists('kmeans_monthly_model.joblib'):
    model_monthly = KMeans(n_clusters=3, init='k-means++')
    model_monthly.fit(data_scaled_monthly_df)
    # Save the trained model to a file
    joblib.dump(model_monthly, 'kmeans_monthly_model.joblib')
else:
    # Load the trained model from the file
    model_monthly = joblib.load('kmeans_monthly_model.joblib')

data_scaled_monthly_df['Cluster'] = model_monthly.predict(data_scaled_monthly_df)


# Function to predict cluster 
def predict_cluster(closing_price):

  scaled_data = scaler.transform([[closing_price]])

  prediction = model.predict(scaled_data)
  if prediction[0]==0:
    regime='Bearish'
  elif prediction[0]==2:
    regime='Consolidated'
  else:
    regime='Bullish'
  return regime

def predict_cluster_weekly(closing_price):

  scaled_data = scaler.transform([[closing_price]])

  prediction = model_weekly.predict(scaled_data)
  if prediction[0]==0:
    regime='Bearish'
  elif prediction[0]==2:
    regime='Consolidated'
  else:
    regime='Bullish'
  return regime

def predict_cluster_monthly(closing_price):

  scaled_data = scaler.transform([[closing_price]])

  prediction = model_monthly.predict(scaled_data)
  if prediction[0]==0:
    regime='Bearish'
  elif prediction[0]==2:
    regime='Consolidated'
  else:
    regime='Bullish'
  return regime

col1,col2,col3=st.columns(3)

with col1:
    # Take input
    closing_price_daily = st.number_input("Enter Daily closing price: ")
with col2:
    # Take input
    closing_price_weekly = st.number_input("Enter Weekly closing price: ")
with col3:
    # Take input
    closing_price_monthly = st.number_input("Enter Monthly closing price: ")


# Predict cluster
if closing_price_daily !=0:
    # Predict cluster
    cluster = predict_cluster(closing_price_daily)
    # Print result
    st.text(f"Closing price {closing_price_daily} belongs to cluster: {cluster}")

if closing_price_weekly !=0:
    # Predict cluster
    cluster = predict_cluster_weekly(closing_price_weekly)
    # Print result
    st.text(f"Closing price {closing_price_weekly} belongs to cluster: {cluster}")

if closing_price_monthly !=0:
    # Predict cluster
    cluster = predict_cluster_monthly(closing_price_monthly)
    # Print result
    st.text(f"Closing price {closing_price_monthly} belongs to cluster: {cluster}")
