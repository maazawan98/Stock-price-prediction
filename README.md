Stock-price-prediction
A machine learning model to predict Reliance stock prices using Linear Regression and Decision Trees.

Project Overview  

Objective:  

The objective of this project is to build a machine learning model to predict the 
next day's stock price of a selected company based on historical stock market 
data. The model leverages time-series forecasting using regression techniques such 
as Linear Regression and Decision Tree Regressor.  

Key Goals:  

Understand and process time-series data for stock price prediction.  
Engineer useful features like lagged prices and percentage changes.  
Build and evaluate regression models: Linear Regression and Decision Tree.  
Compare model performance using Mean Squared Error (MSE) and R² score.  
Visualize trends, predictions, and error margins using Matplotlib and Seaborn.  

Dataset Description  
The dataset is a CSV file containing daily stock price data for a company. Key 
features include:  
Date – The trading date.  
Open – Opening price.  
High – Highest price of the day.  
Low – Lowest price of the day.  
Close – Closing price (target variable).  
Volume – Trading volume.  
For modeling, we focus on predicting the Close price using lag features and 
percentage changes derived from historical data.

Step 1: Import Required Libraries  

We started by importing essential libraries for data manipulation, visualization, and 
modeling:  
import pandas as pd import numpy as np import matplotlib.pyplot as plt import 
seaborn as sns   
from sklearn.linear_model import LinearRegression from sklearn.tree import 
DecisionTreeRegressor from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.model_selection import train_test_split  
These libraries help in data loading (pandas), numerical computation (numpy), data 
visualization (matplotlib, seaborn), and building predictive models (sklearn).  

Step 2: Load and Inspect Dataset  

df = pd.read_csv("RELIANCE.csv") print(df.head()) print(df.info())  
We load the dataset using pandas and inspect the first few rows and column types. 
Initial checks help identify any data type issues or missing values.  

Step 3: Feature Engineering  

Since this is a time-series forecasting problem, we created new features to capture 
historical trends:  
df['Prev_Close'] = df['Close'].shift(1) df['Return'] = df['Close'].pct_change() 
df.dropna(inplace=True) 
Explanation: 
Prev_Close: Yesterday’s closing price helps the model capture temporal patterns.  
Return: Percentage change helps reflect the stock’s volatility and trend.  

Step 4: Exploratory Data Analysis  

We plotted the closing price over time to understand the stock’s movement:  
plt.figure(figsize=(10,5)) plt.plot(df['Close']) plt.title("Closing Price Over Time") 
plt.xlabel("Days") plt.ylabel("Price") plt.show()  
We also plotted the histogram of returns: sns.histplot(df['Return'], bins=50, 
kde=True) plt.title("Daily Return Distribution") plt.show()  
These visualizations highlight the behavior and distribution of the price data, 
including trends and volatility. 

Step 5: Data Preparation for Modeling 

We defined features and labels: X = df[['Prev_Close', 'Return']] y = df['Close']  
Split data into training and testing sets (80% train, 20% test):  
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)  
The shuffle=False ensures that we preserve the time order of data, which is crucial 
in time series forecasting.  

Step 6: Model 1 – Linear Regression  

We trained a simple linear regression model:  
model_lr = LinearRegression() model_lr.fit(X_train, y_train) y_pred_lr = 
model_lr.predict(X_test) Evaluation: mse_lr = mean_squared_error(y_test, 
y_pred_lr) r2_lr = r2_score(y_test, y_pred_lr)  
MSE (Linear Regression): Measures average squared error.  
R² Score: Measures how well predictions approximate actual values.  

Step 7: Model 2 – Decision Tree Regressor  

We trained a tree-based model for potentially capturing nonlinear patterns:  
model_dt = DecisionTreeRegressor(max_depth=5) model_dt.fit(X_train, y_train) 
y_pred_dt = model_dt.predict(X_test) Evaluation: mse_dt = 
mean_squared_error(y_test, y_pred_dt) r2_dt = r2_score(y_test, y_pred_dt)  

Step 8: Visual Comparison of Predictions  

plt.figure(figsize=(12,6)) plt.plot(y_test.values, label="Actual", linewidth=2) 
plt.plot(y_pred_lr, label="Linear Regression", linestyle='--') plt.plot(y_pred_dt, 
label="Decision Tree", linestyle='--') plt.legend() plt.title("Stock Price Predictions 
vs Actual") plt.show()  
This visualization helps compare model performance and error visually over time.  

Step 9: Results & Interpretation Model 

MSE  R² Score  
Linear Regression ~25.13 ~0.97  
Decision Tree  
~15.02 ~0.98  
Both models perform well, but Decision Tree achieves slightly better performance.  
Returns and previous close are effective predictors for next-day stock price.  
Further improvements could involve additional features (technical indicators, 
macroeconomic data).  

Step 10: Conclusion  

The stock prediction models were successfully implemented using historical price 
data. Feature engineering, especially lagged variables and returns, played a key role 
in improving prediction accuracy.  
The Linear Regression model captured linear relationships effectively.  
The Decision Tree model performed better by modeling complex patterns.  
The workflow reflects a complete cycle from raw data to insight extraction and 
predictive modeling.  

THEORY CONCEPTS USED IN TASK 2  

1. Machine Learning Workflow  
Problem Definition: Predict future stock price using past price data.  
Data Collection: CSV file of historical stock prices.  
Data Exploration: Plots and statistics for trends and variance.  
Feature Engineering: Lag variables and percentage returns.  
Model Training: Regression models (Linear and Decision Tree).  
Evaluation: MSE and R² metrics.  
Improvement: Tuning depth for Decision Tree; adding more features.  
Regression vs Classification  
This task is a regression problem (predicting a continuous value) unlike Titanic, 
which was a classification task.  
Evaluation Metrics  
Mean Squared Error (MSE): Penalizes large errors more heavily.  
R² Score: Indicates proportion of variance explained by the model (closer to 1 = 
better).  
Feature Engineering for Time Series  
• Lag Features: Help model sequential dependence in time series. 
• Percentage Return: A normalized way to represent price change.  
Linear Regression  
A statistical method to model the relationship between dependent and independent 
variables. It assumes linearity.  
Decision Tree Regressor  
A tree-based algorithm that splits data into regions for regression. It handles non
linear patterns and is easy to interpret. 
