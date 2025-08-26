import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    df = pd.read_csv("stock data set.csv")  
except FileNotFoundError:
    print("Error: File 'RELIANCE.csv' not found.")
    exit()

df['Prev_Close'] = df['Close'].shift(1)
df['Return'] = df['Close'].pct_change()
df.dropna(inplace=True)

X = df[['Prev_Close', 'Return']]
y = df['Close']


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)


model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

model_dt = DecisionTreeRegressor(max_depth=5)
model_dt.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)
y_pred_dt = model_dt.predict(X_test)

print("\nModel Performance (on test data):")
print(f"Linear Regression MSE: {mean_squared_error(y_test, y_pred_lr):.2f}, R²: {r2_score(y_test, y_pred_lr):.4f}")
print(f"Decision Tree MSE: {mean_squared_error(y_test, y_pred_dt):.2f}, R²: {r2_score(y_test, y_pred_dt):.4f}")

print("\n--- Stock Price Prediction ---")
try:
    prev_close = float(input("Enter the previous day's closing price: "))
    return_pct = float(input("Enter the daily return (e.g., 0.01 for 1%): "))
except ValueError:
    print("Invalid input. Please enter numeric values.")
    exit()

user_input = pd.DataFrame([[prev_close, return_pct]], columns=['Prev_Close', 'Return'])

prediction_lr = model_lr.predict(user_input)[0]
prediction_dt = model_dt.predict(user_input)[0]


print(f"\nPredicted closing price (Linear Regression): ₹{prediction_lr:.2f}")
print(f"Predicted closing price (Decision Tree): ₹{prediction_dt:.2f}")
