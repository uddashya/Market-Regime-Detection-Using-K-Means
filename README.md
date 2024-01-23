## Market-Regime-Detection-Using-K-Means

Project uses simulation, scaling, clustering and prediction to model closing price data, identify different regimes in it, and make predictions about new data points. This provides a quick way to characterize market conditions based on closing prices.

The key benefit is getting a perspective on the market regime through unsupervised learning. The limitations are use of only closing prices and simplicity of the clustering. But overall it demonstrates an end-to-end workflow for analysis and prediction
Stock Market Cluster Analysis
Overview
This application is designed to analyze stock market data, specifically focusing on NIFTY50 index data. It uses KMeans clustering to categorize closing prices into different market regimes such as Bullish, Bearish, or Consolidated. The analysis is performed on daily, weekly, and monthly data.

# Features
Analysis of NIFTY50 stock market data.
KMeans clustering to identify market regimes.
Handling data resampling for different time frames (daily, weekly, monthly).
Predictive functionality to classify new closing price data.

# Usage
Enter daily, weekly, and monthly closing prices on the application's interface to get the market regime predictions.
The application will display which cluster (Bullish, Bearish, Consolidated) the entered closing price belongs to.

## Running the Application

Before running the application, make sure to activate the virtual environment:

```bash
source venv/bin/activate




# Contributing
Contributions are welcome. Please fork the repository and submit pull requests for any enhancements.
