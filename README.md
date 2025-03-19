# NVDA_Complete_LSTM_Forecasts

**A comprehensive Jupyter Notebook for predicting NVIDIA (NVDA) stock metrics using LSTM models.**

This repository contains a Jupyter Notebook, `NVDA_Stock_Predictions_LSTM.ipynb`, that leverages Long Short-Term Memory (LSTM) neural networks to forecast various stock metrics for NVIDIA (NVDA) based on historical daily data. The notebook includes optimized models for predicting closing price, opening price, trading volume, and volatility, using multivariate inputs to maximize accuracy. Ideal for stock market enthusiasts, data scientists, or anyone interested in time series forecasting with machine learning.

## Features
- **Multiple Predictions**:
  - **Closing Price**: Forecast the next day's closing price.
  - **Opening Price**: Predict the next day's opening price with enhanced model architecture.
  - **Trading Volume**: Estimate daily trading volume with an optimized deep LSTM model.
  - **Volatility**: Predict realized volatility (5-day rolling standard deviation of returns) and simpler volatility (High - Low).
- **Multivariate Inputs**: Uses `Close`, `High`, `Low`, `Open`, and `Volume` as features to enrich predictions.
- **Optimized LSTM Models**: Deep architectures with 2-3 layers, increased units (50-150), dropout regularization, and custom learning rates.
- **Training Enhancements**: Early stopping, up to 100 epochs, and fine-tuned Adam optimizer for minimal loss.
- **Visualizations**: Plots for actual vs. predicted values and training/validation loss over epochs.
- **Next-Day Forecasts**: Predicts the next day's value for each metric using the latest data sequence.

## Dataset
- **File**: `"NVDA.csv"` (not included in this repo; user must provide).
- **Format**: Expected columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
- **Source**: Historical daily stock data for NVIDIA (e.g., from Yahoo Finance or similar).

## Requirements
To run the notebook, install the following Python libraries:
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
