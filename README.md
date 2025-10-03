# Predicting AAPL Daily Return Direction Using XGBoost

## Overview
This project uses supervised machine learning to predict whether the next day’s return for Apple Inc. (AAPL) stock will be positive (Up) or negative (Down). The analysis focuses on price-based technical indicators such as lagged closing prices, exponential moving averages (EMA), volume changes, and candlestick-based signals like High–Minus–Low (HML) and Open–Minus–Close (OMC).  
Our goal is to evaluate whether engineered technical features can provide a slight predictive edge in short-term return forecasting.

## Data
Daily historical price data for AAPL (Open, High, Low, Close, Adjusted Close, and Volume) was downloaded from Yahoo Finance starting **October 2, 2015** through September 2025.

### Feature Engineering
- CloseLag1 to CloseLag10: closing price lags capturing short-term momentum and autocorrelation.  
- EMA5, EMA10, EMA20: exponential moving averages to smooth recent price trends.  
- HMLLag and OMCLag: volatility and candlestick-style price differences (high minus low, open minus close).  
- Volume lags and rolling volatility: delayed trading activity effects.  

The target variable is binary: **1 = Up (positive next-day return), 0 = Down (negative next-day return)**.  
All features are standardized using `StandardScaler` to help the gradient boosting model converge.

## Modeling
The model used is **XGBoost (Extreme Gradient Boosting)**.  
Hyperparameter optimization was performed using `RandomizedSearchCV` with **1000 iterations** to maximize classification accuracy.  
Parameters tuned include tree depth, minimum child weight, subsample ratio, learning rate, and number of boosting rounds.  
Time series cross-validation with 5 splits and a 10-day purge gap was applied to avoid look-ahead bias.

### Best Parameters
The best mean cross-validated accuracy achieved was **0.5027**, with the following hyperparameters:
max_depth=13
min_child_weight=2
subsample=0.3244
learning_rate=0.0880
n_estimators=2561

### Final Model Performance
After refitting the model on the full dataset:

- **Accuracy:** 97.55%  
- **AUC:** 0.9973  

Classification report:
          precision    recall  f1-score   support
      0      0.969     0.979     0.974      1164
      1      0.981     0.973     0.977      1330
accuracy                          0.976      2494
macro avg      0.975     0.976     0.975      2494
weighted avg   0.976     0.976     0.976      2494
